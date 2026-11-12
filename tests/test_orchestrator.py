import copy

from agent.llm import LLMReply, RuleBasedLLM
from agent.orchestrator import Agent
from agent.schemas import ToolCall
from tests.conftest import CANNED, BrokenLLM, ScriptedLLM, stub_registry

MANDATORY = (
    "Find the best recent model, identify the dataset version used to train it, check the deployment status, "
    "and explain whether retraining is necessary."
)


def agent(overrides=None, calls=None, llm=None):
    return Agent(registry=stub_registry(overrides, calls), llm=llm or RuleBasedLLM())


def test_tool_selection_for_dataset_question_uses_only_dvc():
    calls = []
    r = agent(calls=calls).run("Which dataset version was used for experiment 25?")
    assert calls == ["get_experiment_dataset"] and "data-v2" in " ".join(r.known_facts)


def test_tool_selection_for_deployment_question():
    calls = []
    agent(calls=calls).run("Why is my prediction API unavailable?")
    assert calls == ["diagnose_deployment"]


def test_mandatory_scenario_uses_all_four_technologies_in_dependency_order():
    calls = []
    r = agent(calls=calls).run(MANDATORY)
    assert calls[0] == "find_best_run" and set(calls) == {
        "find_best_run",
        "get_experiment_dataset",
        "diagnose_deployment",
        "get_deployed_version",
        "get_dag_run_status",
    }
    assert "trigger_training_dag" not in calls
    joined = " ".join(r.known_facts)
    assert all(k in joined for k in ("MLflow:", "DVC:", "Kubernetes:", "Airflow:", "Retraining assessment: not necessary"))
    assert r.uncertainties  # distinguishes facts from uncertainties
    assert all(e.request_id == r.request_id for e in r.evidence)


def test_get_experiment_dataset_receives_run_id_chosen_by_mlflow():
    seen = {}
    reg = stub_registry()
    reg._tools["get_experiment_dataset"].func = lambda a: seen.setdefault("args", a) and CANNED["get_experiment_dataset"]
    Agent(registry=reg, llm=RuleBasedLLM()).run(MANDATORY)
    assert seen["args"].run_id == "b" * 32


def test_agent_does_not_invent_results_when_a_tool_is_unavailable():
    r = agent({"diagnose_deployment": ConnectionError("cluster unreachable"), "get_deployed_version": ConnectionError("x")}).run(MANDATORY)
    k8s_facts = [f for f in r.known_facts if f.startswith("Kubernetes")]
    assert k8s_facts == []  # nothing claimed about the cluster
    assert any("Kubernetes evidence unavailable" in u for u in r.uncertainties)
    assert any("incomplete" in u for u in r.uncertainties)  # retraining verdict is hedged
    assert "healthy=True" not in r.answer and "2/2" not in r.answer


def test_hallucinated_llm_answer_is_discarded_when_every_tool_fails():
    llm = ScriptedLLM(
        [
            LLMReply(tool_calls=[ToolCall(id="1", name="diagnose_deployment", arguments={})]),
            LLMReply(text="All pods are healthy and F1 is 0.99!"),
        ]
    )
    r = agent({"diagnose_deployment": TimeoutError("t")}, llm=llm).run("is the deployment ok?")
    assert "0.99" not in r.answer and "healthy" not in r.answer.lower().replace("unhealthy", "")
    assert "could not verify" in r.answer


def test_llm_prose_is_kept_but_failures_are_appended():
    llm = ScriptedLLM(
        [
            LLMReply(
                tool_calls=[
                    ToolCall(id="1", name="find_best_run", arguments={}),
                    ToolCall(id="2", name="diagnose_deployment", arguments={}),
                ]
            ),
            LLMReply(text="Best run found."),
        ]
    )
    r = agent({"diagnose_deployment": RuntimeError("x")}, llm=llm).run("best model and deployment?")
    assert r.answer.startswith("Best run found.") and "unavailable evidence - diagnose_deployment" in r.answer


def test_retraining_is_not_triggered_automatically():
    calls = []
    agent(calls=calls).run(MANDATORY)
    assert "trigger_training_dag" not in calls


def test_retraining_recommended_when_quality_gate_failed():
    data = copy.deepcopy(CANNED["find_best_run"])
    data["readiness"]["checks"]["quality_gate"] = "failed"
    calls = []
    r = agent({"find_best_run": data}, calls).run(MANDATORY)
    assert any("justified" in f and "quality gate" in f for f in r.known_facts)
    assert any("NOT started automatically" in a for a in r.recommended_actions) and "trigger_training_dag" not in calls


def test_retraining_recommended_when_freshness_policy_exceeded():
    data = copy.deepcopy(CANNED["find_best_run"])
    data["readiness"]["checks"]["age_days"] = 45.0
    r = agent({"find_best_run": data}).run(MANDATORY)
    assert any("freshness policy exceeded" in f for f in r.known_facts)


def test_retraining_recommended_only_with_supplied_degradation_results():
    r = agent().run(MANDATORY, context={"degradation": {"metric": "f1", "value": 0.80}})
    assert any("supplied degradation results" in f for f in r.known_facts)
    r2 = agent().run(MANDATORY, context={"degradation": {"metric": "f1", "value": 0.99}})
    assert not any("justified" in f for f in r2.known_facts)


def test_llm_cannot_self_authorise_state_changes():
    calls = []
    llm = ScriptedLLM(
        [
            LLMReply(tool_calls=[ToolCall(id="1", name="restart_deployment", arguments={"confirm": True, "confirmed": True})]),
            LLMReply(text="restarted"),
        ]
    )
    r = agent(calls=calls, llm=llm).run("restart the deployment")
    assert calls == [] and r.pending_confirmation == ["restart_deployment"] and r.evidence[0].status == "confirmation_required"


def test_caller_confirmation_allows_the_state_change():
    calls = []
    r = agent(calls=calls).run("please restart the deployment", confirm=True)
    assert calls == ["restart_deployment"] and any("rollout restart issued" in f for f in r.known_facts)


def test_training_trigger_request_waits_for_confirmation():
    calls = []
    r = agent(calls=calls).run("Trigger the model training pipeline and report whether execution succeeded.")
    assert calls == [] and r.pending_confirmation == ["trigger_training_dag"]


def test_llm_outage_falls_back_to_offline_planner():
    r = agent(llm=BrokenLLM()).run("Which dataset version was used for experiment 3?")
    assert r.tools_used == ["get_experiment_dataset"] and r.known_facts


def test_unknown_tool_from_llm_is_reported_not_executed():
    llm = ScriptedLLM([LLMReply(tool_calls=[ToolCall(id="1", name="drop_database", arguments={})]), LLMReply(text="")])
    r = agent(llm=llm).run("do something")
    assert r.evidence[0].status == "invalid_request"


def test_unrecognised_request_gets_capability_help_without_tool_calls():
    calls = []
    r = agent(calls=calls).run("tell me a joke about cats")
    assert calls == [] and "DVC" in r.answer


def test_tool_timeout_is_surfaced_in_response(monkeypatch):
    import time

    reg = stub_registry()
    reg._tools["diagnose_deployment"].func = lambda a: time.sleep(2)
    monkeypatch.setenv("TOOL_TIMEOUT_S", "0.2")
    r = Agent(registry=reg, llm=RuleBasedLLM()).run("Why is my prediction API unavailable?")
    assert r.evidence[0].status == "timeout" and "could not verify" in r.answer


def test_malformed_tool_payload_degrades_gracefully_instead_of_failing_the_request():
    r = agent({"find_best_run": {"unexpected": True}}).run("What is the best model by f1?")
    assert any("unexpected format" in u for u in r.uncertainties) and r.evidence[0].status == "ok"
