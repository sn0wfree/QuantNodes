============================= test session starts ==============================
platform linux -- Python 3.11.15, pytest-9.1.1, pluggy-1.6.0
rootdir: /home/ll/Public/QuantNodes
configfile: pyproject.toml
plugins: asyncio-1.4.0, anyio-4.14.1, cov-7.1.0, hypothesis-6.156.1, timeout-2.4.0
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
timeout: 120.0s
timeout method: signal
timeout func_only: False
collected 2002 items

tests/research/factor_test/e2e/test_data_prep_helpers.py .............   [  0%]
tests/research/factor_test/e2e/test_dynamic_dates.py ...                 [  0%]
tests/research/factor_test/e2e/test_e2e_cli_params.py .......            [  1%]
tests/research/factor_test/e2e/test_pipeline_bool_factor.py ............ [  1%]
...                                                                      [  1%]
tests/research/factor_test/e2e/test_run_evolution_e2e_helpers.py ....... [  2%]
...                                                                      [  2%]
tests/research/factor_test/ifind_db/test_batch_size.py ...               [  2%]
tests/research/factor_test/ifind_db/test_fetcher_unit.py ............... [  3%]
                                                                         [  3%]
tests/research/factor_test/ifind_db/test_ifind_database_unit.py ........ [  3%]
.........                                                                [  4%]
tests/research/factor_test/ifind_db/test_ifind_integration.py .....s.    [  4%]
tests/research/factor_test/ifind_db/test_rate_cache_config.py ....       [  4%]
tests/research/factor_test/ifind_db/test_register_route_decorator.py ... [  4%]
..........                                                               [  5%]
tests/research/factor_test/ifind_db/test_risk_registry.py .......        [  5%]
tests/research/factor_test/nodes/test_analysis.py ...................... [  6%]
.......                                                                  [  7%]
tests/research/factor_test/nodes/test_filters.py .............           [  7%]
tests/research/factor_test/nodes/test_load_adjust.py .................   [  8%]
tests/research/factor_test/nodes/test_neutralizer_chain.py ............. [  9%]
........................                                                 [ 10%]
tests/research/factor_test/nodes/test_preprocess.py ..........s......... [ 11%]
........                                                                 [ 11%]
tests/research/factor_test/nodes/test_preprocess_strategies.py ......... [ 12%]
.............................                                            [ 13%]
tests/research/factor_test/nodes/test_score_report.py .................. [ 14%]
......                                                                   [ 14%]
tests/research/factor_test/test_config_builder.py ...............        [ 15%]
tests/research/factor_test/test_config_settings.py ..................... [ 16%]
.......                                                                  [ 17%]
tests/research/factor_test/test_evolution_config_defaults.py ....        [ 17%]
tests/research/factor_test/test_high_hardcoded_fixes.py ...........      [ 17%]
tests/research/factor_test/test_node_configs_extra.py .................. [ 18%]
.....                                                                    [ 19%]
tests/research/factor_test/test_r4_microcleanup.py ...                   [ 19%]
tests/research/factor_test/test_register_node_config.py ...........      [ 19%]
tests/research/factor_test/utils/test_constants_overrides.py ..........  [ 20%]
tests/research/factor_test/utils/test_date_perf_metrics.py ..........    [ 20%]
tests/research/factor_test/utils/test_date_utils_edge_cases.py ......... [ 21%]
...........................                                              [ 22%]
tests/research/factor_test/utils/test_evaluation_annual_days.py ....     [ 22%]
tests/research/factor_test/utils/test_hypothesis_properties.py ......... [ 23%]
                                                                         [ 23%]
tests/research/factor_test/utils/test_performance_metrics_full.py ...... [ 23%]
.............                                                            [ 24%]
tests/research/factor_test/utils/test_safe_load.py .........             [ 24%]
tests/research/test_akshare_data.py ........                             [ 24%]
tests/research/test_ast_compiler.py .......                              [ 25%]
tests/research/test_ast_complexity.py .......                            [ 25%]
tests/research/test_ast_extractor.py .....                               [ 25%]
tests/research/test_ast_nodes.py ..............                          [ 26%]
tests/research/test_auto_research.py ................................... [ 28%]
.................                                                        [ 29%]
tests/research/test_backtest.py ......                                   [ 29%]
tests/research/test_clickhouse_data.py ........                          [ 29%]
tests/research/test_codegen_utils_more.py .............                  [ 30%]
tests/research/test_contracts.py ...............                         [ 31%]
tests/research/test_core.py .............................                [ 32%]
tests/research/test_data_loader_edges.py ............................... [ 34%]
.............                                                            [ 34%]
tests/research/test_data_source_integration.py ..........                [ 35%]
tests/research/test_date_utils_edges.py ............................     [ 36%]
tests/research/test_defer.py .................                           [ 37%]
tests/research/test_e2e_smoke.py ....................................... [ 39%]
......................                                                   [ 40%]
tests/research/test_equity.py .....                                      [ 41%]
tests/research/test_error_categorizer.py ..........                      [ 41%]
tests/research/test_extract.py ................                          [ 42%]
tests/research/test_extract_factors.py ......                            [ 42%]
tests/research/test_extract_paper.py ..............                      [ 43%]
tests/research/test_factor_backtest.py ............                      [ 43%]
tests/research/test_factor_compiler.py .....                             [ 44%]
tests/research/test_factor_compiler_react.py ............s               [ 44%]
tests/research/test_factor_extractor_more.py .....                       [ 45%]
tests/research/test_factor_library.py .......................            [ 46%]
tests/research/test_factor_value_store.py .....                          [ 46%]
tests/research/test_file_loaders.py .................                    [ 47%]
tests/research/test_ifind_data.py ..............                         [ 48%]
tests/research/test_ifind_fetcher_edges.py ............................. [ 49%]
.........                                                                [ 49%]
tests/research/test_invariants.py .......................                [ 51%]
tests/research/test_l4_hypothesis_sync.py ..                             [ 51%]
tests/research/test_l5_orchestrator.py ...........                       [ 51%]
tests/research/test_l5_reflection.py ...............                     [ 52%]
tests/research/test_l5_reverse_factor_scoring.py .....                   [ 52%]
tests/research/test_l5_stability_oos.py .................                [ 53%]
tests/research/test_l5_validation.py .........                           [ 53%]
tests/research/test_llm_extraction_config.py .....                       [ 54%]
tests/research/test_llm_factory.py ......                                [ 54%]
tests/research/test_log_decorator.py .............                       [ 55%]
tests/research/test_metrics_more.py ........                             [ 55%]
tests/research/test_multi_factor_extraction.py .........                 [ 56%]
tests/research/test_node_configs.py .................................... [ 57%]
........................................................................ [ 61%]
......                                                                   [ 61%]
tests/research/test_nodes_edges.py ............                          [ 62%]
tests/research/test_orchestrator.py .......s.....                        [ 62%]
tests/research/test_orchestrator_helpers.py .................            [ 63%]
tests/research/test_p0_backtest_fixes.py ............                    [ 64%]
tests/research/test_paper_api.py ..................                      [ 65%]
tests/research/test_parquet_and_formula.py ..........                    [ 65%]
tests/research/test_paths.py ............                                [ 66%]
tests/research/test_performance_metrics.py ..........                    [ 66%]
tests/research/test_pipeline_equivalence.py ..........                   [ 67%]
tests/research/test_pipeline_runner_helpers.py ..........                [ 67%]
tests/research/test_pipeline_smoke.py ...                                [ 68%]
tests/research/test_plan_saver.py .............                          [ 68%]
tests/research/test_planner_helpers.py ...........................       [ 70%]
tests/research/test_planner_more.py ........                             [ 70%]
tests/research/test_preview_more.py ......                               [ 70%]
tests/research/test_prompts.py .......................                   [ 71%]
tests/research/test_quant_wiki.py ........                               [ 72%]
tests/research/test_quantnodes_adapter.py ........                       [ 72%]
tests/research/test_quantnodes_repro_more.py .......                     [ 73%]
tests/research/test_react_self_repair_e2e.py ..........                  [ 73%]
tests/research/test_report_reproducer.py .......................         [ 74%]
tests/research/test_reporting.py ................................        [ 76%]
tests/research/test_repro_config.py ............                         [ 76%]
tests/research/test_repro_integration.py ........                        [ 77%]
tests/research/test_reproduction_api.py .........                        [ 77%]
tests/research/test_retry.py ........................                    [ 78%]
tests/research/test_retry_integration.py .........                       [ 79%]
tests/research/test_router.py ............                               [ 80%]
tests/research/test_run.py ..........                                    [ 80%]
tests/research/test_run_id.py .....                                      [ 80%]
tests/research/test_runlog.py ..................                         [ 81%]
tests/research/test_schemas.py .........                                 [ 82%]
tests/research/test_section_detector.py ......                           [ 82%]
tests/research/test_section_detector_helpers.py ....................     [ 83%]
tests/research/test_self_repairing.py ..........                         [ 83%]
tests/research/test_sessions.py ............                             [ 84%]
tests/research/test_signal_source.py ..............................      [ 86%]
tests/research/test_sink.py .......................                      [ 87%]
tests/research/test_stage0_ingest.py ..................                  [ 88%]
tests/research/test_strategies.py .......                                [ 88%]
tests/research/test_strategy_api.py sssss                                [ 88%]
tests/research/test_success_rate.py ..................                   [ 89%]
tests/research/test_telemetry.py .....                                   [ 89%]
tests/research/test_track_a.py ......                                    [ 90%]
tests/research/test_track_b_adaptive_helpers.py ........................ [ 91%]
.................                                                        [ 92%]
tests/research/test_track_b_checkpoint.py ................               [ 92%]
tests/research/test_track_b_hybrid.py ............................       [ 94%]
tests/research/test_track_b_multiturn.py .............sss                [ 95%]
tests/research/test_universe.py .......................                  [ 96%]
tests/research/test_utils.py .................                           [ 97%]
tests/research/test_validator_preview.py ...............ssss...          [ 98%]
tests/research/test_wiki.py ...................................          [100%]

=============================== warnings summary ===============================
QuantNodes/research/_legacy_3c/factor_evaluator.py:33
  /home/ll/Public/QuantNodes/QuantNodes/research/_legacy_3c/factor_evaluator.py:33: DeprecationWarning: QuantNodes.research.factor_miner 已弃用 (DeprecationWarning)。请迁移到 QuantNodes.research.quant_alpha.operator_vocab.OperatorVocab。
    from QuantNodes.research._legacy_3c.factor_miner import FactorCandidate

QuantNodes/research/_legacy_3c/__init__.py:28
  /home/ll/Public/QuantNodes/QuantNodes/research/_legacy_3c/__init__.py:28: DeprecationWarning: QuantNodes.research.factor_evaluator 已弃用 (DeprecationWarning)。请迁移到 QuantNodes.research.quant_alpha.operator_vocab.OperatorVocab (162 算子 + 修复 3 个 latent bug + 完整元数据)。Phase B (v2.9+): 本类变 thin wrapper。Phase C (v3.0): 归档到 _legacy_3c/。
    from .factor_evaluator import (

QuantNodes/research/_legacy_3c/__init__.py:39
  /home/ll/Public/QuantNodes/QuantNodes/research/_legacy_3c/__init__.py:39: DeprecationWarning: QuantNodes.research.mcts_search 已弃用 (DeprecationWarning)。M2 PR 将提供新实现 QuantNodes.research.quant_alpha.mcts.MCTSSearch。
    from .mcts_search import (

QuantNodes/research/_legacy_3c/__init__.py:43
  /home/ll/Public/QuantNodes/QuantNodes/research/_legacy_3c/__init__.py:43: DeprecationWarning: QuantNodes.research.auto_researcher 已弃用 (DeprecationWarning)。M5+ PR 将提供新实现 QuantNodes.research.quant_alpha.AutoResearcher。
    from .auto_researcher import (

tests/research/factor_test/nodes/test_analysis.py::TestICAnalyzerNode::test_constant_factor_ic_is_nan
  /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/pandas/core/nanops.py:1673: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
    return spearmanr(a, b)[0]

tests/research/factor_test/nodes/test_analysis.py::TestICAnalyzerNode::test_constant_factor_ic_is_nan
tests/research/factor_test/nodes/test_analysis.py::TestICAnalyzerNode::test_min_group_size_filters_sparse_dates
  /home/ll/Public/QuantNodes/QuantNodes/research/factor_test/nodes/ic_analyzer_node.py:88: RuntimeWarning: invalid value encountered in sqrt
    ic.mean() / ic.std(ddof=1) * np.sqrt(ic.notna().sum() - 1)

tests/research/factor_test/nodes/test_analysis.py::TestICAnalyzerNode::test_constant_factor_ic_is_nan
tests/research/factor_test/nodes/test_analysis.py::TestICAnalyzerNode::test_min_group_size_filters_sparse_dates
  /home/ll/Public/QuantNodes/QuantNodes/research/factor_test/nodes/ic_analyzer_node.py:97: RuntimeWarning: invalid value encountered in sqrt
    rank_ic.mean() / rank_ic.std(ddof=1) * np.sqrt(rank_ic.notna().sum() - 1)

tests/research/test_extract_factors.py::test_build_factor_pages
  /home/ll/Public/QuantNodes/tests/research/test_extract_factors.py:83: DeprecationWarning: build_factor_pages() is deprecated; use factor_library.write_factor_yaml()
    pages = build_factor_pages(factors, "test-001")

tests/research/test_extract_factors.py::test_build_factor_pages_slug
  /home/ll/Public/QuantNodes/tests/research/test_extract_factors.py:94: DeprecationWarning: build_factor_pages() is deprecated; use factor_library.write_factor_yaml()
    pages = build_factor_pages(factors, "test-002")

tests/research/test_extract_factors.py::test_build_factor_pages_multiple
  /home/ll/Public/QuantNodes/tests/research/test_extract_factors.py:103: DeprecationWarning: build_factor_pages() is deprecated; use factor_library.write_factor_yaml()
    pages = build_factor_pages(factors, "test-003")

tests/research/test_factor_compiler_react.py::test_react_retries_on_extract_failure
  /home/ll/Public/QuantNodes/tests/research/test_factor_compiler_react.py:128: DeprecationWarning: compile_to_code_react is deprecated. Use llmwikify.kernel.agent.generate_factor_code_sync instead.
    result = compile_to_code_react(

tests/research/test_factor_compiler_react.py::test_react_retries_on_syntax_error
  /home/ll/Public/QuantNodes/tests/research/test_factor_compiler_react.py:174: DeprecationWarning: compile_to_code_react is deprecated. Use llmwikify.kernel.agent.generate_factor_code_sync instead.
    result = compile_to_code_react(

tests/research/test_factor_compiler_react.py::test_react_retries_on_execution_error
  /home/ll/Public/QuantNodes/tests/research/test_factor_compiler_react.py:210: DeprecationWarning: compile_to_code_react is deprecated. Use llmwikify.kernel.agent.generate_factor_code_sync instead.
    result = compile_to_code_react(

tests/research/test_factor_compiler_react.py::test_react_exhausts_rounds
  /home/ll/Public/QuantNodes/tests/research/test_factor_compiler_react.py:245: DeprecationWarning: compile_to_code_react is deprecated. Use llmwikify.kernel.agent.generate_factor_code_sync instead.
    result = compile_to_code_react(

tests/research/test_factor_compiler_react.py::test_react_succeeds_on_first_try
  /home/ll/Public/QuantNodes/tests/research/test_factor_compiler_react.py:273: DeprecationWarning: compile_to_code_react is deprecated. Use llmwikify.kernel.agent.generate_factor_code_sync instead.
    result = compile_to_code_react(

tests/research/test_factor_compiler_react.py::test_telemetry_records_self_repair_events
  /home/ll/Public/QuantNodes/tests/research/test_factor_compiler_react.py:310: DeprecationWarning: compile_to_code_react is deprecated. Use llmwikify.kernel.agent.generate_factor_code_sync instead.
    compile_to_code_react(

tests/research/test_factor_compiler_react.py::test_progress_callback_called_per_step
  /home/ll/Public/QuantNodes/tests/research/test_factor_compiler_react.py:348: DeprecationWarning: compile_to_code_react is deprecated. Use llmwikify.kernel.agent.generate_factor_code_sync instead.
    result = compile_to_code_react(

tests/research/test_llm_factory.py::TestBuildDefaultClient::test_missing_config_raises
  /home/ll/Public/QuantNodes/tests/research/test_llm_factory.py:61: DeprecationWarning: reproduction.common.llm_factory.build_default_client is deprecated; use QuantNodes.research.common.llm.client.build_llm_client instead. This wrapper will be removed in a future release.
    lf.build_default_client()

tests/research/test_llm_factory.py::TestBuildDefaultClient::test_builds_client_with_valid_config
  /home/ll/Public/QuantNodes/tests/research/test_llm_factory.py:80: DeprecationWarning: reproduction.common.llm_factory.build_default_client is deprecated; use QuantNodes.research.common.llm.client.build_llm_client instead. This wrapper will be removed in a future release.
    client = lf.build_default_client()

tests/research/test_llm_factory.py::TestBuildDefaultClient::test_model_override
  /home/ll/Public/QuantNodes/tests/research/test_llm_factory.py:103: DeprecationWarning: reproduction.common.llm_factory.build_default_client is deprecated; use QuantNodes.research.common.llm.client.build_llm_client instead. This wrapper will be removed in a future release.
    client = lf.build_default_client(model="override-model")

tests/research/test_orchestrator.py::TestOrchestratorEnd2End::test_defer_section_detector_continues_with_no_sections
tests/research/test_orchestrator.py::TestOrchestratorEnd2End::test_no_defer_no_queue
tests/research/test_orchestrator.py::TestOrchestratorEnd2End::test_no_defer_no_queue
tests/research/test_orchestrator.py::TestOrchestratorEnd2End::test_no_defer_no_queue
  /home/ll/Public/QuantNodes/QuantNodes/research/paper_understanding/llm_extraction/orchestrator.py:256: DeprecationWarning: reproduction.common.llm_factory.build_default_client is deprecated; use QuantNodes.research.common.llm.client.build_llm_client instead. This wrapper will be removed in a future release.
    client = llm_client or build_default_client()

tests/research/test_paper_api.py::test_start_returns_session_id
  /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/fastapi/testclient.py:1: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
    from starlette.testclient import TestClient as TestClient  # noqa

tests/research/test_paper_api.py: 20 warnings
  /home/ll/llmwikify/src/llmwikify/reproduction/pipeline/workflow.py:185: DeprecationWarning: reproduction.common.llm_factory.build_default_client is deprecated; use llmwikify.kernel.quant.llm_client.build_llm_client instead. This wrapper will be removed in a future release.
    return build_default_client()

tests/research/test_react_self_repair_e2e.py::TestReactWithMockLLM::test_compile_with_good_llm
  /home/ll/Public/QuantNodes/tests/research/test_react_self_repair_e2e.py:58: DeprecationWarning: compile_to_code_react is deprecated. Use llmwikify.kernel.agent.generate_factor_code_sync instead.
    result = fcr.compile_to_code_react(

tests/research/test_react_self_repair_e2e.py::TestReactWithMockLLM::test_compile_with_bad_llm_safety_error
  /home/ll/Public/QuantNodes/tests/research/test_react_self_repair_e2e.py:76: DeprecationWarning: compile_to_code_react is deprecated. Use llmwikify.kernel.agent.generate_factor_code_sync instead.
    result = fcr.compile_to_code_react(

tests/research/test_react_self_repair_e2e.py::TestReactWithMockLLM::test_compile_repair_loop
  /home/ll/Public/QuantNodes/tests/research/test_react_self_repair_e2e.py:101: DeprecationWarning: compile_to_code_react is deprecated. Use llmwikify.kernel.agent.generate_factor_code_sync instead.
    result = fcr.compile_to_code_react(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
========== 1986 passed, 16 skipped, 50 warnings in 173.12s (0:02:53) ===========
