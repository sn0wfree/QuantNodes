============================= test session starts ==============================
platform linux -- Python 3.11.15, pytest-9.1.1, pluggy-1.6.0
rootdir: /home/ll/Public/QuantNodes
configfile: pyproject.toml
plugins: asyncio-1.4.0, anyio-4.14.1, cov-7.1.0, hypothesis-6.156.1, timeout-2.4.0
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
timeout: 120.0s
timeout method: signal
timeout func_only: False
collected 1865 items

tests/research/factor_test/e2e/test_data_prep_helpers.py .............   [  0%]
tests/research/factor_test/e2e/test_dynamic_dates.py ...                 [  0%]
tests/research/factor_test/e2e/test_e2e_cli_params.py .......            [  1%]
tests/research/factor_test/e2e/test_pipeline_bool_factor.py ............ [  1%]
...                                                                      [  2%]
tests/research/factor_test/e2e/test_run_evolution_e2e_helpers.py ....... [  2%]
...                                                                      [  2%]
tests/research/factor_test/ifind_db/test_batch_size.py ...               [  2%]
tests/research/factor_test/ifind_db/test_fetcher_unit.py ............... [  3%]
                                                                         [  3%]
tests/research/factor_test/ifind_db/test_ifind_database_unit.py ........ [  3%]
.........                                                                [  4%]
tests/research/factor_test/ifind_db/test_ifind_integration.py .....s.    [  4%]
tests/research/factor_test/ifind_db/test_rate_cache_config.py ....       [  5%]
tests/research/factor_test/ifind_db/test_register_route_decorator.py ... [  5%]
..........                                                               [  5%]
tests/research/factor_test/ifind_db/test_risk_registry.py .......        [  6%]
tests/research/factor_test/nodes/test_analysis.py ...................... [  7%]
.......                                                                  [  7%]
tests/research/factor_test/nodes/test_filters.py .............           [  8%]
tests/research/factor_test/nodes/test_load_adjust.py .................   [  9%]
tests/research/factor_test/nodes/test_neutralizer_chain.py ............. [  9%]
........................                                                 [ 11%]
tests/research/factor_test/nodes/test_preprocess.py ..........s......... [ 12%]
........                                                                 [ 12%]
tests/research/factor_test/nodes/test_preprocess_strategies.py ......... [ 13%]
.............................                                            [ 14%]
tests/research/factor_test/nodes/test_score_report.py .................. [ 15%]
......                                                                   [ 16%]
tests/research/factor_test/test_config_builder.py ...............        [ 16%]
tests/research/factor_test/test_config_settings.py ..................... [ 18%]
.......                                                                  [ 18%]
tests/research/factor_test/test_evolution_config_defaults.py ....        [ 18%]
tests/research/factor_test/test_high_hardcoded_fixes.py ...........      [ 19%]
tests/research/factor_test/test_node_configs_extra.py .................. [ 20%]
.....                                                                    [ 20%]
tests/research/factor_test/test_r4_microcleanup.py ...                   [ 20%]
tests/research/factor_test/test_register_node_config.py ...........      [ 21%]
tests/research/factor_test/utils/test_constants_overrides.py ..........  [ 21%]
tests/research/factor_test/utils/test_date_perf_metrics.py ..........    [ 22%]
tests/research/factor_test/utils/test_date_utils_edge_cases.py ......... [ 22%]
...........................                                              [ 24%]
tests/research/factor_test/utils/test_evaluation_annual_days.py ....     [ 24%]
tests/research/factor_test/utils/test_hypothesis_properties.py ......... [ 24%]
                                                                         [ 24%]
tests/research/factor_test/utils/test_performance_metrics_full.py ...... [ 25%]
.............                                                            [ 25%]
tests/research/factor_test/utils/test_safe_load.py .........             [ 26%]
tests/research/test_akshare_data.py ........                             [ 26%]
tests/research/test_ast_compiler.py .......                              [ 27%]
tests/research/test_ast_complexity.py .......                            [ 27%]
tests/research/test_ast_extractor.py .....                               [ 27%]
tests/research/test_ast_nodes.py ..............                          [ 28%]
tests/research/test_auto_research.py ...............................EEEE [ 30%]
...........EEEEEE                                                        [ 31%]
tests/research/test_backtest.py ......                                   [ 31%]
tests/research/test_clickhouse_data.py ........                          [ 32%]
tests/research/test_codegen_utils_more.py .............                  [ 32%]
tests/research/test_contracts.py ...............                         [ 33%]
tests/research/test_data_loader_edges.py ............................... [ 35%]
.............                                                            [ 35%]
tests/research/test_data_source_integration.py ..........                [ 36%]
tests/research/test_date_utils_edges.py ............................     [ 38%]
tests/research/test_defer.py .................                           [ 38%]
tests/research/test_e2e_smoke.py ....................................... [ 41%]
......................                                                   [ 42%]
tests/research/test_equity.py .....                                      [ 42%]
tests/research/test_error_categorizer.py ..........                      [ 43%]
tests/research/test_extract.py ................                          [ 43%]
tests/research/test_extract_factors.py ......                            [ 44%]
tests/research/test_extract_paper.py ..............                      [ 44%]
tests/research/test_factor_backtest.py ............                      [ 45%]
tests/research/test_factor_compiler.py .....                             [ 45%]
tests/research/test_factor_compiler_react.py ............s               [ 46%]
tests/research/test_factor_extractor_more.py .....                       [ 46%]
tests/research/test_factor_library.py .......................            [ 48%]
tests/research/test_factor_value_store.py .....                          [ 48%]
tests/research/test_file_loaders.py .................                    [ 49%]
tests/research/test_ifind_data.py ..............                         [ 49%]
tests/research/test_ifind_fetcher_edges.py ............................. [ 51%]
.........                                                                [ 52%]
tests/research/test_invariants.py .......................                [ 53%]
tests/research/test_l4_hypothesis_sync.py ..                             [ 53%]
tests/research/test_l5_orchestrator.py ...........                       [ 53%]
tests/research/test_l5_reflection.py ...............                     [ 54%]
tests/research/test_l5_reverse_factor_scoring.py .....                   [ 55%]
tests/research/test_l5_stability_oos.py .................                [ 55%]
tests/research/test_l5_validation.py .........                           [ 56%]
tests/research/test_llm_extraction_config.py .....                       [ 56%]
tests/research/test_llm_factory.py ....F.                                [ 56%]
tests/research/test_log_decorator.py .............                       [ 57%]
tests/research/test_metrics_more.py ........                             [ 58%]
tests/research/test_multi_factor_extraction.py .........                 [ 58%]
tests/research/test_node_configs.py .................................... [ 60%]
........................................................................ [ 64%]
......                                                                   [ 64%]
tests/research/test_nodes_edges.py ............                          [ 65%]
tests/research/test_orchestrator.py .......s..F.F                        [ 66%]
tests/research/test_orchestrator_helpers.py .................            [ 66%]
tests/research/test_p0_backtest_fixes.py ............                    [ 67%]
tests/research/test_paper_api.py EEEEEEEEEEEEEEEEEE                      [ 68%]
tests/research/test_parquet_and_formula.py ..........                    [ 69%]
tests/research/test_paths.py ............                                [ 69%]
tests/research/test_performance_metrics.py ..........                    [ 70%]
tests/research/test_pipeline_equivalence.py ..........                   [ 70%]
tests/research/test_pipeline_runner_helpers.py ..........                [ 71%]
tests/research/test_pipeline_smoke.py ...                                [ 71%]
tests/research/test_plan_saver.py .............                          [ 72%]
tests/research/test_planner_helpers.py ...........................       [ 73%]
tests/research/test_planner_more.py ........                             [ 74%]
tests/research/test_preview_more.py ......                               [ 74%]
tests/research/test_quant_wiki.py ........                               [ 74%]
tests/research/test_quantnodes_adapter.py ........                       [ 75%]
tests/research/test_quantnodes_repro_more.py .......                     [ 75%]
tests/research/test_react_self_repair_e2e.py ..........                  [ 76%]
tests/research/test_report_reproducer.py ...EEEEEEEEEEEEEEEEEEEE         [ 77%]
tests/research/test_repro_config.py ............                         [ 78%]
tests/research/test_repro_integration.py ........                        [ 78%]
tests/research/test_reproduction_api.py EEEEEEEEE                        [ 78%]
tests/research/test_retry.py ........................                    [ 80%]
tests/research/test_retry_integration.py ....FFFFF                       [ 80%]
tests/research/test_router.py ............                               [ 81%]
tests/research/test_run.py ..........                                    [ 81%]
tests/research/test_run_id.py .....                                      [ 82%]
tests/research/test_runlog.py ..................                         [ 83%]
tests/research/test_schemas.py .........                                 [ 83%]
tests/research/test_section_detector.py ......                           [ 83%]
tests/research/test_section_detector_helpers.py ....................     [ 85%]
tests/research/test_self_repairing.py ..........                         [ 85%]
tests/research/test_sessions.py ............                             [ 86%]
tests/research/test_stage0_ingest.py ..................                  [ 87%]
tests/research/test_strategies.py .......                                [ 87%]
tests/research/test_strategy_api.py sssss                                [ 87%]
tests/research/test_success_rate.py ..................                   [ 88%]
tests/research/test_telemetry.py .....                                   [ 89%]
tests/research/test_track_a.py ......                                    [ 89%]
tests/research/test_track_b_adaptive_helpers.py .................FFFFFFF [ 90%]
.................                                                        [ 91%]
tests/research/test_track_b_checkpoint.py ................               [ 92%]
tests/research/test_track_b_hybrid.py ................FF...FFF....       [ 93%]
tests/research/test_track_b_multiturn.py ......FFFFFFFsss                [ 94%]
tests/research/test_universe.py .......................                  [ 96%]
tests/research/test_utils.py .................                           [ 96%]
tests/research/test_validator_preview.py ...............ssss...          [ 98%]
tests/research/test_wiki.py FEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE          [100%]

==================================== ERRORS ====================================
_____________ ERROR at setup of TestAutoResearcher.test_run_basic ______________

    @pytest.fixture
    def tmp_wiki():
        from QuantNodes.research.wiki import init_factor_wiki
        d = tempfile.mkdtemp()
        wiki_path = str(Path(d) / "test_wiki")
>       init_factor_wiki(wiki_path)

tests/research/test_auto_research.py:47: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888ea7c5b50>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
________ ERROR at setup of TestAutoResearcher.test_run_generates_report ________

    @pytest.fixture
    def tmp_wiki():
        from QuantNodes.research.wiki import init_factor_wiki
        d = tempfile.mkdtemp()
        wiki_path = str(Path(d) / "test_wiki")
>       init_factor_wiki(wiki_path)

tests/research/test_auto_research.py:47: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888d1f71210>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
_________ ERROR at setup of TestAutoResearcher.test_mine_single_factor _________

    @pytest.fixture
    def tmp_wiki():
        from QuantNodes.research.wiki import init_factor_wiki
        d = tempfile.mkdtemp()
        wiki_path = str(Path(d) / "test_wiki")
>       init_factor_wiki(wiki_path)

tests/research/test_auto_research.py:47: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888d293a9d0>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
___________ ERROR at setup of TestAutoResearcher.test_store_to_wiki ____________

    @pytest.fixture
    def tmp_wiki():
        from QuantNodes.research.wiki import init_factor_wiki
        d = tempfile.mkdtemp()
        wiki_path = str(Path(d) / "test_wiki")
>       init_factor_wiki(wiki_path)

tests/research/test_auto_research.py:47: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888d29dc550>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
______ ERROR at setup of TestAutoResearcherEdge.test_run_with_empty_data _______

    @pytest.fixture
    def tmp_wiki():
        from QuantNodes.research.wiki import init_factor_wiki
        d = tempfile.mkdtemp()
        wiki_path = str(Path(d) / "test_wiki")
>       init_factor_wiki(wiki_path)

tests/research/test_auto_research.py:47: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888ea338510>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
___ ERROR at setup of TestAutoResearcherEdge.test_run_with_max_factors_zero ____

    @pytest.fixture
    def tmp_wiki():
        from QuantNodes.research.wiki import init_factor_wiki
        d = tempfile.mkdtemp()
        wiki_path = str(Path(d) / "test_wiki")
>       init_factor_wiki(wiki_path)

tests/research/test_auto_research.py:47: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888d1f22c50>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
__ ERROR at setup of TestAutoResearcherEdge.test_run_with_custom_eval_config ___

    @pytest.fixture
    def tmp_wiki():
        from QuantNodes.research.wiki import init_factor_wiki
        d = tempfile.mkdtemp()
        wiki_path = str(Path(d) / "test_wiki")
>       init_factor_wiki(wiki_path)

tests/research/test_auto_research.py:47: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888d2966350>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
_______ ERROR at setup of TestAutoResearcherEdge.test_run_use_mcts_flag ________

    @pytest.fixture
    def tmp_wiki():
        from QuantNodes.research.wiki import init_factor_wiki
        d = tempfile.mkdtemp()
        wiki_path = str(Path(d) / "test_wiki")
>       init_factor_wiki(wiki_path)

tests/research/test_auto_research.py:47: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888d2981a90>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
_ ERROR at setup of TestAutoResearcherEdge.test_mine_single_factor_invalid_formula _

    @pytest.fixture
    def tmp_wiki():
        from QuantNodes.research.wiki import init_factor_wiki
        d = tempfile.mkdtemp()
        wiki_path = str(Path(d) / "test_wiki")
>       init_factor_wiki(wiki_path)

tests/research/test_auto_research.py:47: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888d1f992d0>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
_ ERROR at setup of TestAutoResearcherEdge.test_mine_single_factor_stores_to_wiki _

    @pytest.fixture
    def tmp_wiki():
        from QuantNodes.research.wiki import init_factor_wiki
        d = tempfile.mkdtemp()
        wiki_path = str(Path(d) / "test_wiki")
>       init_factor_wiki(wiki_path)

tests/research/test_auto_research.py:47: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888ea77d590>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
_______________ ERROR at setup of test_start_returns_session_id ________________
file /home/ll/Public/QuantNodes/tests/research/test_paper_api.py, line 23
  def test_start_returns_session_id(paper_client):
E       fixture 'paper_client' not found
>       available fixtures: _asyncio_loop_factory, _class_scoped_runner, _function_scoped_runner, _module_scoped_runner, _package_scoped_runner, _session_scoped_runner, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, capteesys, cov, doctest_namespace, eval_data, event_loop_policy, factor_evaluator, factor_miner, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, market_data_df, market_data_pdf, mock_llm_client, mock_wiki_pages, monitor_db, monkeypatch, no_cover, polars_df, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_df, sample_df_with_null, sample_factor, sample_pdf_text, subtests, temp_csv_file, temp_csv_file_with_null, temp_duckdb_db, temp_parquet_file, temp_sqlite_db, temp_yaml_config, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, wiki_path, wiki_proxy
>       use 'pytest --fixtures [testpath]' for help on them.

/home/ll/Public/QuantNodes/tests/research/test_paper_api.py:23
_________________ ERROR at setup of test_start_default_wiki_id _________________
file /home/ll/Public/QuantNodes/tests/research/test_paper_api.py, line 33
  def test_start_default_wiki_id(paper_client):
E       fixture 'paper_client' not found
>       available fixtures: _asyncio_loop_factory, _class_scoped_runner, _function_scoped_runner, _module_scoped_runner, _package_scoped_runner, _session_scoped_runner, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, capteesys, cov, doctest_namespace, eval_data, event_loop_policy, factor_evaluator, factor_miner, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, market_data_df, market_data_pdf, mock_llm_client, mock_wiki_pages, monitor_db, monkeypatch, no_cover, polars_df, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_df, sample_df_with_null, sample_factor, sample_pdf_text, subtests, temp_csv_file, temp_csv_file_with_null, temp_duckdb_db, temp_parquet_file, temp_sqlite_db, temp_yaml_config, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, wiki_path, wiki_proxy
>       use 'pytest --fixtures [testpath]' for help on them.

/home/ll/Public/QuantNodes/tests/research/test_paper_api.py:33
________________ ERROR at setup of test_start_explicit_wiki_id _________________
file /home/ll/Public/QuantNodes/tests/research/test_paper_api.py, line 43
  def test_start_explicit_wiki_id(paper_client):
E       fixture 'paper_client' not found
>       available fixtures: _asyncio_loop_factory, _class_scoped_runner, _function_scoped_runner, _module_scoped_runner, _package_scoped_runner, _session_scoped_runner, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, capteesys, cov, doctest_namespace, eval_data, event_loop_policy, factor_evaluator, factor_miner, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, market_data_df, market_data_pdf, mock_llm_client, mock_wiki_pages, monitor_db, monkeypatch, no_cover, polars_df, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_df, sample_df_with_null, sample_factor, sample_pdf_text, subtests, temp_csv_file, temp_csv_file_with_null, temp_duckdb_db, temp_parquet_file, temp_sqlite_db, temp_yaml_config, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, wiki_path, wiki_proxy
>       use 'pytest --fixtures [testpath]' for help on them.

/home/ll/Public/QuantNodes/tests/research/test_paper_api.py:43
_______________ ERROR at setup of test_start_invalid_source_type _______________
file /home/ll/Public/QuantNodes/tests/research/test_paper_api.py, line 51
  def test_start_invalid_source_type(paper_client):
E       fixture 'paper_client' not found
>       available fixtures: _asyncio_loop_factory, _class_scoped_runner, _function_scoped_runner, _module_scoped_runner, _package_scoped_runner, _session_scoped_runner, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, capteesys, cov, doctest_namespace, eval_data, event_loop_policy, factor_evaluator, factor_miner, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, market_data_df, market_data_pdf, mock_llm_client, mock_wiki_pages, monitor_db, monkeypatch, no_cover, polars_df, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_df, sample_df_with_null, sample_factor, sample_pdf_text, subtests, temp_csv_file, temp_csv_file_with_null, temp_duckdb_db, temp_parquet_file, temp_sqlite_db, temp_yaml_config, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, wiki_path, wiki_proxy
>       use 'pytest --fixtures [testpath]' for help on them.

/home/ll/Public/QuantNodes/tests/research/test_paper_api.py:51
______________________ ERROR at setup of test_list_empty _______________________
file /home/ll/Public/QuantNodes/tests/research/test_paper_api.py, line 60
  def test_list_empty(paper_client):
E       fixture 'paper_client' not found
>       available fixtures: _asyncio_loop_factory, _class_scoped_runner, _function_scoped_runner, _module_scoped_runner, _package_scoped_runner, _session_scoped_runner, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, capteesys, cov, doctest_namespace, eval_data, event_loop_policy, factor_evaluator, factor_miner, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, market_data_df, market_data_pdf, mock_llm_client, mock_wiki_pages, monitor_db, monkeypatch, no_cover, polars_df, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_df, sample_df_with_null, sample_factor, sample_pdf_text, subtests, temp_csv_file, temp_csv_file_with_null, temp_duckdb_db, temp_parquet_file, temp_sqlite_db, temp_yaml_config, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, wiki_path, wiki_proxy
>       use 'pytest --fixtures [testpath]' for help on them.

/home/ll/Public/QuantNodes/tests/research/test_paper_api.py:60
__________________ ERROR at setup of test_list_with_sessions ___________________
file /home/ll/Public/QuantNodes/tests/research/test_paper_api.py, line 67
  def test_list_with_sessions(paper_client):
E       fixture 'paper_client' not found
>       available fixtures: _asyncio_loop_factory, _class_scoped_runner, _function_scoped_runner, _module_scoped_runner, _package_scoped_runner, _session_scoped_runner, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, capteesys, cov, doctest_namespace, eval_data, event_loop_policy, factor_evaluator, factor_miner, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, market_data_df, market_data_pdf, mock_llm_client, mock_wiki_pages, monitor_db, monkeypatch, no_cover, polars_df, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_df, sample_df_with_null, sample_factor, sample_pdf_text, subtests, temp_csv_file, temp_csv_file_with_null, temp_duckdb_db, temp_parquet_file, temp_sqlite_db, temp_yaml_config, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, wiki_path, wiki_proxy
>       use 'pytest --fixtures [testpath]' for help on them.

/home/ll/Public/QuantNodes/tests/research/test_paper_api.py:67
__________________ ERROR at setup of test_list_filter_status ___________________
file /home/ll/Public/QuantNodes/tests/research/test_paper_api.py, line 77
  def test_list_filter_status(paper_client):
E       fixture 'paper_client' not found
>       available fixtures: _asyncio_loop_factory, _class_scoped_runner, _function_scoped_runner, _module_scoped_runner, _package_scoped_runner, _session_scoped_runner, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, capteesys, cov, doctest_namespace, eval_data, event_loop_policy, factor_evaluator, factor_miner, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, market_data_df, market_data_pdf, mock_llm_client, mock_wiki_pages, monitor_db, monkeypatch, no_cover, polars_df, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_df, sample_df_with_null, sample_factor, sample_pdf_text, subtests, temp_csv_file, temp_csv_file_with_null, temp_duckdb_db, temp_parquet_file, temp_sqlite_db, temp_yaml_config, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, wiki_path, wiki_proxy
>       use 'pytest --fixtures [testpath]' for help on them.

/home/ll/Public/QuantNodes/tests/research/test_paper_api.py:77
____________________ ERROR at setup of test_list_raw_empty _____________________
file /home/ll/Public/QuantNodes/tests/research/test_paper_api.py, line 97
  def test_list_raw_empty(paper_client):
E       fixture 'paper_client' not found
>       available fixtures: _asyncio_loop_factory, _class_scoped_runner, _function_scoped_runner, _module_scoped_runner, _package_scoped_runner, _session_scoped_runner, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, capteesys, cov, doctest_namespace, eval_data, event_loop_policy, factor_evaluator, factor_miner, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, market_data_df, market_data_pdf, mock_llm_client, mock_wiki_pages, monitor_db, monkeypatch, no_cover, polars_df, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_df, sample_df_with_null, sample_factor, sample_pdf_text, subtests, temp_csv_file, temp_csv_file_with_null, temp_duckdb_db, temp_parquet_file, temp_sqlite_db, temp_yaml_config, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, wiki_path, wiki_proxy
>       use 'pytest --fixtures [testpath]' for help on them.

/home/ll/Public/QuantNodes/tests/research/test_paper_api.py:97
_______________________ ERROR at setup of test_list_raw ________________________
file /home/ll/Public/QuantNodes/tests/research/test_paper_api.py, line 104
  def test_list_raw(paper_client):
E       fixture 'paper_client' not found
>       available fixtures: _asyncio_loop_factory, _class_scoped_runner, _function_scoped_runner, _module_scoped_runner, _package_scoped_runner, _session_scoped_runner, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, capteesys, cov, doctest_namespace, eval_data, event_loop_policy, factor_evaluator, factor_miner, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, market_data_df, market_data_pdf, mock_llm_client, mock_wiki_pages, monitor_db, monkeypatch, no_cover, polars_df, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_df, sample_df_with_null, sample_factor, sample_pdf_text, subtests, temp_csv_file, temp_csv_file_with_null, temp_duckdb_db, temp_parquet_file, temp_sqlite_db, temp_yaml_config, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, wiki_path, wiki_proxy
>       use 'pytest --fixtures [testpath]' for help on them.

/home/ll/Public/QuantNodes/tests/research/test_paper_api.py:104
______________________ ERROR at setup of test_upload_pdf _______________________
file /home/ll/Public/QuantNodes/tests/research/test_paper_api.py, line 123
  def test_upload_pdf(paper_client):
E       fixture 'paper_client' not found
>       available fixtures: _asyncio_loop_factory, _class_scoped_runner, _function_scoped_runner, _module_scoped_runner, _package_scoped_runner, _session_scoped_runner, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, capteesys, cov, doctest_namespace, eval_data, event_loop_policy, factor_evaluator, factor_miner, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, market_data_df, market_data_pdf, mock_llm_client, mock_wiki_pages, monitor_db, monkeypatch, no_cover, polars_df, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_df, sample_df_with_null, sample_factor, sample_pdf_text, subtests, temp_csv_file, temp_csv_file_with_null, temp_duckdb_db, temp_parquet_file, temp_sqlite_db, temp_yaml_config, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, wiki_path, wiki_proxy
>       use 'pytest --fixtures [testpath]' for help on them.

/home/ll/Public/QuantNodes/tests/research/test_paper_api.py:123
________________ ERROR at setup of test_upload_rejects_non_pdf _________________
file /home/ll/Public/QuantNodes/tests/research/test_paper_api.py, line 137
  def test_upload_rejects_non_pdf(paper_client):
E       fixture 'paper_client' not found
>       available fixtures: _asyncio_loop_factory, _class_scoped_runner, _function_scoped_runner, _module_scoped_runner, _package_scoped_runner, _session_scoped_runner, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, capteesys, cov, doctest_namespace, eval_data, event_loop_policy, factor_evaluator, factor_miner, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, market_data_df, market_data_pdf, mock_llm_client, mock_wiki_pages, monitor_db, monkeypatch, no_cover, polars_df, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_df, sample_df_with_null, sample_factor, sample_pdf_text, subtests, temp_csv_file, temp_csv_file_with_null, temp_duckdb_db, temp_parquet_file, temp_sqlite_db, temp_yaml_config, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, wiki_path, wiki_proxy
>       use 'pytest --fixtures [testpath]' for help on them.

/home/ll/Public/QuantNodes/tests/research/test_paper_api.py:137
_________________ ERROR at setup of test_upload_rejects_empty __________________
file /home/ll/Public/QuantNodes/tests/research/test_paper_api.py, line 147
  def test_upload_rejects_empty(paper_client):
E       fixture 'paper_client' not found
>       available fixtures: _asyncio_loop_factory, _class_scoped_runner, _function_scoped_runner, _module_scoped_runner, _package_scoped_runner, _session_scoped_runner, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, capteesys, cov, doctest_namespace, eval_data, event_loop_policy, factor_evaluator, factor_miner, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, market_data_df, market_data_pdf, mock_llm_client, mock_wiki_pages, monitor_db, monkeypatch, no_cover, polars_df, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_df, sample_df_with_null, sample_factor, sample_pdf_text, subtests, temp_csv_file, temp_csv_file_with_null, temp_duckdb_db, temp_parquet_file, temp_sqlite_db, temp_yaml_config, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, wiki_path, wiki_proxy
>       use 'pytest --fixtures [testpath]' for help on them.

/home/ll/Public/QuantNodes/tests/research/test_paper_api.py:147
___________ ERROR at setup of test_status_returns_session_and_events ___________
file /home/ll/Public/QuantNodes/tests/research/test_paper_api.py, line 160
  def test_status_returns_session_and_events(paper_client):
E       fixture 'paper_client' not found
>       available fixtures: _asyncio_loop_factory, _class_scoped_runner, _function_scoped_runner, _module_scoped_runner, _package_scoped_runner, _session_scoped_runner, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, capteesys, cov, doctest_namespace, eval_data, event_loop_policy, factor_evaluator, factor_miner, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, market_data_df, market_data_pdf, mock_llm_client, mock_wiki_pages, monitor_db, monkeypatch, no_cover, polars_df, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_df, sample_df_with_null, sample_factor, sample_pdf_text, subtests, temp_csv_file, temp_csv_file_with_null, temp_duckdb_db, temp_parquet_file, temp_sqlite_db, temp_yaml_config, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, wiki_path, wiki_proxy
>       use 'pytest --fixtures [testpath]' for help on them.

/home/ll/Public/QuantNodes/tests/research/test_paper_api.py:160
___________________ ERROR at setup of test_status_not_found ____________________
file /home/ll/Public/QuantNodes/tests/research/test_paper_api.py, line 174
  def test_status_not_found(paper_client):
E       fixture 'paper_client' not found
>       available fixtures: _asyncio_loop_factory, _class_scoped_runner, _function_scoped_runner, _module_scoped_runner, _package_scoped_runner, _session_scoped_runner, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, capteesys, cov, doctest_namespace, eval_data, event_loop_policy, factor_evaluator, factor_miner, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, market_data_df, market_data_pdf, mock_llm_client, mock_wiki_pages, monitor_db, monkeypatch, no_cover, polars_df, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_df, sample_df_with_null, sample_factor, sample_pdf_text, subtests, temp_csv_file, temp_csv_file_with_null, temp_duckdb_db, temp_parquet_file, temp_sqlite_db, temp_yaml_config, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, wiki_path, wiki_proxy
>       use 'pytest --fixtures [testpath]' for help on them.

/home/ll/Public/QuantNodes/tests/research/test_paper_api.py:174
____________________ ERROR at setup of test_legacy_paper_id ____________________
file /home/ll/Public/QuantNodes/tests/research/test_paper_api.py, line 183
  def test_legacy_paper_id(paper_client):
E       fixture 'paper_client' not found
>       available fixtures: _asyncio_loop_factory, _class_scoped_runner, _function_scoped_runner, _module_scoped_runner, _package_scoped_runner, _session_scoped_runner, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, capteesys, cov, doctest_namespace, eval_data, event_loop_policy, factor_evaluator, factor_miner, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, market_data_df, market_data_pdf, mock_llm_client, mock_wiki_pages, monitor_db, monkeypatch, no_cover, polars_df, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_df, sample_df_with_null, sample_factor, sample_pdf_text, subtests, temp_csv_file, temp_csv_file_with_null, temp_duckdb_db, temp_parquet_file, temp_sqlite_db, temp_yaml_config, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, wiki_path, wiki_proxy
>       use 'pytest --fixtures [testpath]' for help on them.

/home/ll/Public/QuantNodes/tests/research/test_paper_api.py:183
___________________ ERROR at setup of test_legacy_artifacts ____________________
file /home/ll/Public/QuantNodes/tests/research/test_paper_api.py, line 196
  def test_legacy_artifacts(paper_client):
E       fixture 'paper_client' not found
>       available fixtures: _asyncio_loop_factory, _class_scoped_runner, _function_scoped_runner, _module_scoped_runner, _package_scoped_runner, _session_scoped_runner, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, capteesys, cov, doctest_namespace, eval_data, event_loop_policy, factor_evaluator, factor_miner, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, market_data_df, market_data_pdf, mock_llm_client, mock_wiki_pages, monitor_db, monkeypatch, no_cover, polars_df, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_df, sample_df_with_null, sample_factor, sample_pdf_text, subtests, temp_csv_file, temp_csv_file_with_null, temp_duckdb_db, temp_parquet_file, temp_sqlite_db, temp_yaml_config, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, wiki_path, wiki_proxy
>       use 'pytest --fixtures [testpath]' for help on them.

/home/ll/Public/QuantNodes/tests/research/test_paper_api.py:196
____________________ ERROR at setup of test_delete_session _____________________
file /home/ll/Public/QuantNodes/tests/research/test_paper_api.py, line 208
  def test_delete_session(paper_client):
E       fixture 'paper_client' not found
>       available fixtures: _asyncio_loop_factory, _class_scoped_runner, _function_scoped_runner, _module_scoped_runner, _package_scoped_runner, _session_scoped_runner, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, capteesys, cov, doctest_namespace, eval_data, event_loop_policy, factor_evaluator, factor_miner, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, market_data_df, market_data_pdf, mock_llm_client, mock_wiki_pages, monitor_db, monkeypatch, no_cover, polars_df, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_df, sample_df_with_null, sample_factor, sample_pdf_text, subtests, temp_csv_file, temp_csv_file_with_null, temp_duckdb_db, temp_parquet_file, temp_sqlite_db, temp_yaml_config, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, wiki_path, wiki_proxy
>       use 'pytest --fixtures [testpath]' for help on them.

/home/ll/Public/QuantNodes/tests/research/test_paper_api.py:208
___________________ ERROR at setup of test_delete_not_found ____________________
file /home/ll/Public/QuantNodes/tests/research/test_paper_api.py, line 221
  def test_delete_not_found(paper_client):
E       fixture 'paper_client' not found
>       available fixtures: _asyncio_loop_factory, _class_scoped_runner, _function_scoped_runner, _module_scoped_runner, _package_scoped_runner, _session_scoped_runner, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, capteesys, cov, doctest_namespace, eval_data, event_loop_policy, factor_evaluator, factor_miner, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, market_data_df, market_data_pdf, mock_llm_client, mock_wiki_pages, monitor_db, monkeypatch, no_cover, polars_df, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_df, sample_df_with_null, sample_factor, sample_pdf_text, subtests, temp_csv_file, temp_csv_file_with_null, temp_duckdb_db, temp_parquet_file, temp_sqlite_db, temp_yaml_config, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, wiki_path, wiki_proxy
>       use 'pytest --fixtures [testpath]' for help on them.

/home/ll/Public/QuantNodes/tests/research/test_paper_api.py:221
____________ ERROR at setup of TestParsePdf.test_parse_pdf_fallback ____________

    @pytest.fixture
    def tmp_wiki():
        d = tempfile.mkdtemp()
        wiki_path = str(Path(d) / "test_wiki")
>       init_factor_wiki(wiki_path)

tests/research/test_report_reproducer.py:48: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888d1f15b50>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
_____ ERROR at setup of TestRuleBasedExtract.test_extract_formula_patterns _____

    @pytest.fixture
    def tmp_wiki():
        d = tempfile.mkdtemp()
        wiki_path = str(Path(d) / "test_wiki")
>       init_factor_wiki(wiki_path)

tests/research/test_report_reproducer.py:48: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888d29809d0>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
_________ ERROR at setup of TestRuleBasedExtract.test_extract_no_match _________

    @pytest.fixture
    def tmp_wiki():
        d = tempfile.mkdtemp()
        wiki_path = str(Path(d) / "test_wiki")
>       init_factor_wiki(wiki_path)

tests/research/test_report_reproducer.py:48: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888d01b9190>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
__________ ERROR at setup of TestRuleBasedExtract.test_extract_dedup ___________

    @pytest.fixture
    def tmp_wiki():
        d = tempfile.mkdtemp()
        wiki_path = str(Path(d) / "test_wiki")
>       init_factor_wiki(wiki_path)

tests/research/test_report_reproducer.py:48: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888eb579150>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
__________ ERROR at setup of TestLLMExtract.test_llm_extract_success ___________

    @pytest.fixture
    def tmp_wiki():
        d = tempfile.mkdtemp()
        wiki_path = str(Path(d) / "test_wiki")
>       init_factor_wiki(wiki_path)

tests/research/test_report_reproducer.py:48: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888d018b790>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
_____ ERROR at setup of TestLLMExtract.test_llm_extract_fallback_on_error ______

    @pytest.fixture
    def tmp_wiki():
        d = tempfile.mkdtemp()
        wiki_path = str(Path(d) / "test_wiki")
>       init_factor_wiki(wiki_path)

tests/research/test_report_reproducer.py:48: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x78893a459590>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
_________ ERROR at setup of TestVerifyFactor.test_verify_factor_valid __________

    @pytest.fixture
    def tmp_wiki():
        d = tempfile.mkdtemp()
        wiki_path = str(Path(d) / "test_wiki")
>       init_factor_wiki(wiki_path)

tests/research/test_report_reproducer.py:48: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888eb548d50>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
__________ ERROR at setup of TestVerifyFactor.test_verify_no_formula ___________

    @pytest.fixture
    def tmp_wiki():
        d = tempfile.mkdtemp()
        wiki_path = str(Path(d) / "test_wiki")
>       init_factor_wiki(wiki_path)

tests/research/test_report_reproducer.py:48: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888d0149190>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
________ ERROR at setup of TestVerifyFactor.test_verify_invalid_formula ________

    @pytest.fixture
    def tmp_wiki():
        d = tempfile.mkdtemp()
        wiki_path = str(Path(d) / "test_wiki")
>       init_factor_wiki(wiki_path)

tests/research/test_report_reproducer.py:48: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888d054cb50>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
_________ ERROR at setup of TestWikiStorage.test_store_verified_factor _________

    @pytest.fixture
    def tmp_wiki():
        d = tempfile.mkdtemp()
        wiki_path = str(Path(d) / "test_wiki")
>       init_factor_wiki(wiki_path)

tests/research/test_report_reproducer.py:48: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888d0f4d3d0>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
__________ ERROR at setup of TestWikiStorage.test_store_pending_logic __________

    @pytest.fixture
    def tmp_wiki():
        d = tempfile.mkdtemp()
        wiki_path = str(Path(d) / "test_wiki")
>       init_factor_wiki(wiki_path)

tests/research/test_report_reproducer.py:48: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888d1ff8050>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
_________ ERROR at setup of TestReportGeneration.test_generate_report __________

    @pytest.fixture
    def tmp_wiki():
        d = tempfile.mkdtemp()
        wiki_path = str(Path(d) / "test_wiki")
>       init_factor_wiki(wiki_path)

tests/research/test_report_reproducer.py:48: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888d0f0e5d0>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
_______ ERROR at setup of TestProcessE2E.test_process_with_rule_extract ________

    @pytest.fixture
    def tmp_wiki():
        d = tempfile.mkdtemp()
        wiki_path = str(Path(d) / "test_wiki")
>       init_factor_wiki(wiki_path)

tests/research/test_report_reproducer.py:48: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888d0189f10>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
__________ ERROR at setup of TestProcessE2E.test_process_without_data __________

    @pytest.fixture
    def tmp_wiki():
        d = tempfile.mkdtemp()
        wiki_path = str(Path(d) / "test_wiki")
>       init_factor_wiki(wiki_path)

tests/research/test_report_reproducer.py:48: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888d1ff3510>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
_ ERROR at setup of TestExtractLogicFromText.test_extract_logic_from_text_public _

    @pytest.fixture
    def tmp_wiki():
        d = tempfile.mkdtemp()
        wiki_path = str(Path(d) / "test_wiki")
>       init_factor_wiki(wiki_path)

tests/research/test_report_reproducer.py:48: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888e9f4fe50>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
_ ERROR at setup of TestExtractLogicFromText.test_extract_logic_from_text_empty _

    @pytest.fixture
    def tmp_wiki():
        d = tempfile.mkdtemp()
        wiki_path = str(Path(d) / "test_wiki")
>       init_factor_wiki(wiki_path)

tests/research/test_report_reproducer.py:48: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888d1ff2e50>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
_ ERROR at setup of TestExtractLogicFromText.test_extract_logic_from_text_with_llm _

    @pytest.fixture
    def tmp_wiki():
        d = tempfile.mkdtemp()
        wiki_path = str(Path(d) / "test_wiki")
>       init_factor_wiki(wiki_path)

tests/research/test_report_reproducer.py:48: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888ea3e9310>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
_ ERROR at setup of TestReproductionReportDataclass.test_reproduction_report_fields _

    @pytest.fixture
    def tmp_wiki():
        d = tempfile.mkdtemp()
        wiki_path = str(Path(d) / "test_wiki")
>       init_factor_wiki(wiki_path)

tests/research/test_report_reproducer.py:48: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888e9f4bcd0>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
__ ERROR at setup of TestProcessWithInvalidPdf.test_process_invalid_pdf_path ___

    @pytest.fixture
    def tmp_wiki():
        d = tempfile.mkdtemp()
        wiki_path = str(Path(d) / "test_wiki")
>       init_factor_wiki(wiki_path)

tests/research/test_report_reproducer.py:48: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888d0138150>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
_ ERROR at setup of TestProcessWithDataNoFormulas.test_process_with_data_no_formulas _

    @pytest.fixture
    def tmp_wiki():
        d = tempfile.mkdtemp()
        wiki_path = str(Path(d) / "test_wiki")
>       init_factor_wiki(wiki_path)

tests/research/test_report_reproducer.py:48: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888d01b4210>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
_______________ ERROR at setup of test_start_returns_session_id ________________
file /home/ll/Public/QuantNodes/tests/research/test_reproduction_api.py, line 23
  def test_start_returns_session_id(repro_client):
E       fixture 'repro_client' not found
>       available fixtures: _asyncio_loop_factory, _class_scoped_runner, _function_scoped_runner, _module_scoped_runner, _package_scoped_runner, _session_scoped_runner, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, capteesys, cov, doctest_namespace, eval_data, event_loop_policy, factor_evaluator, factor_miner, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, market_data_df, market_data_pdf, mock_llm_client, mock_wiki_pages, monitor_db, monkeypatch, no_cover, polars_df, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_df, sample_df_with_null, sample_factor, sample_pdf_text, subtests, temp_csv_file, temp_csv_file_with_null, temp_duckdb_db, temp_parquet_file, temp_sqlite_db, temp_yaml_config, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, wiki_path, wiki_proxy
>       use 'pytest --fixtures [testpath]' for help on them.

/home/ll/Public/QuantNodes/tests/research/test_reproduction_api.py:23
_________________ ERROR at setup of test_start_default_wiki_id _________________
file /home/ll/Public/QuantNodes/tests/research/test_reproduction_api.py, line 32
  def test_start_default_wiki_id(repro_client):
E       fixture 'repro_client' not found
>       available fixtures: _asyncio_loop_factory, _class_scoped_runner, _function_scoped_runner, _module_scoped_runner, _package_scoped_runner, _session_scoped_runner, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, capteesys, cov, doctest_namespace, eval_data, event_loop_policy, factor_evaluator, factor_miner, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, market_data_df, market_data_pdf, mock_llm_client, mock_wiki_pages, monitor_db, monkeypatch, no_cover, polars_df, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_df, sample_df_with_null, sample_factor, sample_pdf_text, subtests, temp_csv_file, temp_csv_file_with_null, temp_duckdb_db, temp_parquet_file, temp_sqlite_db, temp_yaml_config, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, wiki_path, wiki_proxy
>       use 'pytest --fixtures [testpath]' for help on them.

/home/ll/Public/QuantNodes/tests/research/test_reproduction_api.py:32
______________________ ERROR at setup of test_list_empty _______________________
file /home/ll/Public/QuantNodes/tests/research/test_reproduction_api.py, line 45
  def test_list_empty(repro_client):
E       fixture 'repro_client' not found
>       available fixtures: _asyncio_loop_factory, _class_scoped_runner, _function_scoped_runner, _module_scoped_runner, _package_scoped_runner, _session_scoped_runner, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, capteesys, cov, doctest_namespace, eval_data, event_loop_policy, factor_evaluator, factor_miner, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, market_data_df, market_data_pdf, mock_llm_client, mock_wiki_pages, monitor_db, monkeypatch, no_cover, polars_df, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_df, sample_df_with_null, sample_factor, sample_pdf_text, subtests, temp_csv_file, temp_csv_file_with_null, temp_duckdb_db, temp_parquet_file, temp_sqlite_db, temp_yaml_config, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, wiki_path, wiki_proxy
>       use 'pytest --fixtures [testpath]' for help on them.

/home/ll/Public/QuantNodes/tests/research/test_reproduction_api.py:45
__________________ ERROR at setup of test_list_with_sessions ___________________
file /home/ll/Public/QuantNodes/tests/research/test_reproduction_api.py, line 52
  def test_list_with_sessions(repro_client):
E       fixture 'repro_client' not found
>       available fixtures: _asyncio_loop_factory, _class_scoped_runner, _function_scoped_runner, _module_scoped_runner, _package_scoped_runner, _session_scoped_runner, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, capteesys, cov, doctest_namespace, eval_data, event_loop_policy, factor_evaluator, factor_miner, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, market_data_df, market_data_pdf, mock_llm_client, mock_wiki_pages, monitor_db, monkeypatch, no_cover, polars_df, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_df, sample_df_with_null, sample_factor, sample_pdf_text, subtests, temp_csv_file, temp_csv_file_with_null, temp_duckdb_db, temp_parquet_file, temp_sqlite_db, temp_yaml_config, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, wiki_path, wiki_proxy
>       use 'pytest --fixtures [testpath]' for help on them.

/home/ll/Public/QuantNodes/tests/research/test_reproduction_api.py:52
______________________ ERROR at setup of test_get_session ______________________
file /home/ll/Public/QuantNodes/tests/research/test_reproduction_api.py, line 64
  def test_get_session(repro_client):
E       fixture 'repro_client' not found
>       available fixtures: _asyncio_loop_factory, _class_scoped_runner, _function_scoped_runner, _module_scoped_runner, _package_scoped_runner, _session_scoped_runner, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, capteesys, cov, doctest_namespace, eval_data, event_loop_policy, factor_evaluator, factor_miner, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, market_data_df, market_data_pdf, mock_llm_client, mock_wiki_pages, monitor_db, monkeypatch, no_cover, polars_df, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_df, sample_df_with_null, sample_factor, sample_pdf_text, subtests, temp_csv_file, temp_csv_file_with_null, temp_duckdb_db, temp_parquet_file, temp_sqlite_db, temp_yaml_config, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, wiki_path, wiki_proxy
>       use 'pytest --fixtures [testpath]' for help on them.

/home/ll/Public/QuantNodes/tests/research/test_reproduction_api.py:64
_____________________ ERROR at setup of test_get_not_found _____________________
file /home/ll/Public/QuantNodes/tests/research/test_reproduction_api.py, line 76
  def test_get_not_found(repro_client):
E       fixture 'repro_client' not found
>       available fixtures: _asyncio_loop_factory, _class_scoped_runner, _function_scoped_runner, _module_scoped_runner, _package_scoped_runner, _session_scoped_runner, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, capteesys, cov, doctest_namespace, eval_data, event_loop_policy, factor_evaluator, factor_miner, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, market_data_df, market_data_pdf, mock_llm_client, mock_wiki_pages, monitor_db, monkeypatch, no_cover, polars_df, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_df, sample_df_with_null, sample_factor, sample_pdf_text, subtests, temp_csv_file, temp_csv_file_with_null, temp_duckdb_db, temp_parquet_file, temp_sqlite_db, temp_yaml_config, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, wiki_path, wiki_proxy
>       use 'pytest --fixtures [testpath]' for help on them.

/home/ll/Public/QuantNodes/tests/research/test_reproduction_api.py:76
____________________ ERROR at setup of test_artifacts_empty ____________________
file /home/ll/Public/QuantNodes/tests/research/test_reproduction_api.py, line 85
  def test_artifacts_empty(repro_client):
E       fixture 'repro_client' not found
>       available fixtures: _asyncio_loop_factory, _class_scoped_runner, _function_scoped_runner, _module_scoped_runner, _package_scoped_runner, _session_scoped_runner, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, capteesys, cov, doctest_namespace, eval_data, event_loop_policy, factor_evaluator, factor_miner, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, market_data_df, market_data_pdf, mock_llm_client, mock_wiki_pages, monitor_db, monkeypatch, no_cover, polars_df, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_df, sample_df_with_null, sample_factor, sample_pdf_text, subtests, temp_csv_file, temp_csv_file_with_null, temp_duckdb_db, temp_parquet_file, temp_sqlite_db, temp_yaml_config, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, wiki_path, wiki_proxy
>       use 'pytest --fixtures [testpath]' for help on them.

/home/ll/Public/QuantNodes/tests/research/test_reproduction_api.py:85
____________________ ERROR at setup of test_delete_session _____________________
file /home/ll/Public/QuantNodes/tests/research/test_reproduction_api.py, line 99
  def test_delete_session(repro_client):
E       fixture 'repro_client' not found
>       available fixtures: _asyncio_loop_factory, _class_scoped_runner, _function_scoped_runner, _module_scoped_runner, _package_scoped_runner, _session_scoped_runner, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, capteesys, cov, doctest_namespace, eval_data, event_loop_policy, factor_evaluator, factor_miner, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, market_data_df, market_data_pdf, mock_llm_client, mock_wiki_pages, monitor_db, monkeypatch, no_cover, polars_df, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_df, sample_df_with_null, sample_factor, sample_pdf_text, subtests, temp_csv_file, temp_csv_file_with_null, temp_duckdb_db, temp_parquet_file, temp_sqlite_db, temp_yaml_config, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, wiki_path, wiki_proxy
>       use 'pytest --fixtures [testpath]' for help on them.

/home/ll/Public/QuantNodes/tests/research/test_reproduction_api.py:99
___________________ ERROR at setup of test_delete_not_found ____________________
file /home/ll/Public/QuantNodes/tests/research/test_reproduction_api.py, line 111
  def test_delete_not_found(repro_client):
E       fixture 'repro_client' not found
>       available fixtures: _asyncio_loop_factory, _class_scoped_runner, _function_scoped_runner, _module_scoped_runner, _package_scoped_runner, _session_scoped_runner, anyio_backend, anyio_backend_name, anyio_backend_options, cache, capfd, capfdbinary, caplog, capsys, capsysbinary, capteesys, cov, doctest_namespace, eval_data, event_loop_policy, factor_evaluator, factor_miner, free_tcp_port, free_tcp_port_factory, free_udp_port, free_udp_port_factory, market_data_df, market_data_pdf, mock_llm_client, mock_wiki_pages, monitor_db, monkeypatch, no_cover, polars_df, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, sample_df, sample_df_with_null, sample_factor, sample_pdf_text, subtests, temp_csv_file, temp_csv_file_with_null, temp_duckdb_db, temp_parquet_file, temp_sqlite_db, temp_yaml_config, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory, unused_tcp_port, unused_tcp_port_factory, unused_udp_port, unused_udp_port_factory, wiki_path, wiki_proxy
>       use 'pytest --fixtures [testpath]' for help on them.

/home/ll/Public/QuantNodes/tests/research/test_reproduction_api.py:111
_________ ERROR at setup of TestWikiFactorProxyCRUD.test_store_factor __________

    @pytest.fixture
    def tmp_wiki_path():
        """临时 wiki 目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            wiki_path = os.path.join(tmpdir, "test_wiki")
>           init_factor_wiki(wiki_path)

tests/research/test_wiki.py:21: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888ea382290>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
__________ ERROR at setup of TestWikiFactorProxyCRUD.test_get_factor ___________

    @pytest.fixture
    def tmp_wiki_path():
        """临时 wiki 目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            wiki_path = os.path.join(tmpdir, "test_wiki")
>           init_factor_wiki(wiki_path)

tests/research/test_wiki.py:21: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888d1f2a010>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
_____ ERROR at setup of TestWikiFactorProxyCRUD.test_get_factor_not_found ______

    @pytest.fixture
    def tmp_wiki_path():
        """临时 wiki 目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            wiki_path = os.path.join(tmpdir, "test_wiki")
>           init_factor_wiki(wiki_path)

tests/research/test_wiki.py:21: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888b95c7c90>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
________ ERROR at setup of TestWikiFactorProxyCRUD.test_search_factors _________

    @pytest.fixture
    def tmp_wiki_path():
        """临时 wiki 目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            wiki_path = os.path.join(tmpdir, "test_wiki")
>           init_factor_wiki(wiki_path)

tests/research/test_wiki.py:21: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x78889a9def10>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
_________ ERROR at setup of TestWikiFactorProxyCRUD.test_list_factors __________

    @pytest.fixture
    def tmp_wiki_path():
        """临时 wiki 目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            wiki_path = os.path.join(tmpdir, "test_wiki")
>           init_factor_wiki(wiki_path)

tests/research/test_wiki.py:21: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x78889a90add0>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
___ ERROR at setup of TestWikiFactorProxyCRUD.test_list_factors_with_filter ____

    @pytest.fixture
    def tmp_wiki_path():
        """临时 wiki 目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            wiki_path = os.path.join(tmpdir, "test_wiki")
>           init_factor_wiki(wiki_path)

tests/research/test_wiki.py:21: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x78889a96a190>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
_ ERROR at setup of TestWikiFactorProxyCRUD.test_list_factors_with_tags_filter _

    @pytest.fixture
    def tmp_wiki_path():
        """临时 wiki 目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            wiki_path = os.path.join(tmpdir, "test_wiki")
>           init_factor_wiki(wiki_path)

tests/research/test_wiki.py:21: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888a00ed7d0>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
_ ERROR at setup of TestWikiFactorProxyCRUD.test_list_factors_with_category_filter _

    @pytest.fixture
    def tmp_wiki_path():
        """临时 wiki 目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            wiki_path = os.path.join(tmpdir, "test_wiki")
>           init_factor_wiki(wiki_path)

tests/research/test_wiki.py:21: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888b81b0f90>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
_________ ERROR at setup of TestWikiFactorProxyCRUD.test_update_factor _________

    @pytest.fixture
    def tmp_wiki_path():
        """临时 wiki 目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            wiki_path = os.path.join(tmpdir, "test_wiki")
>           init_factor_wiki(wiki_path)

tests/research/test_wiki.py:21: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x78889a99b790>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
____ ERROR at setup of TestWikiFactorProxyCRUD.test_update_factor_not_found ____

    @pytest.fixture
    def tmp_wiki_path():
        """临时 wiki 目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            wiki_path = os.path.join(tmpdir, "test_wiki")
>           init_factor_wiki(wiki_path)

tests/research/test_wiki.py:21: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888a00d6bd0>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
_________ ERROR at setup of TestWikiFactorProxyCRUD.test_delete_factor _________

    @pytest.fixture
    def tmp_wiki_path():
        """临时 wiki 目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            wiki_path = os.path.join(tmpdir, "test_wiki")
>           init_factor_wiki(wiki_path)

tests/research/test_wiki.py:21: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888b81aa350>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
____ ERROR at setup of TestWikiFactorProxyCRUD.test_delete_factor_not_found ____

    @pytest.fixture
    def tmp_wiki_path():
        """临时 wiki 目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            wiki_path = os.path.join(tmpdir, "test_wiki")
>           init_factor_wiki(wiki_path)

tests/research/test_wiki.py:21: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x78889a995a50>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
_____________ ERROR at setup of TestWikiLogicCRUD.test_store_logic _____________

    @pytest.fixture
    def tmp_wiki_path():
        """临时 wiki 目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            wiki_path = os.path.join(tmpdir, "test_wiki")
>           init_factor_wiki(wiki_path)

tests/research/test_wiki.py:21: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888b81e1390>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
______________ ERROR at setup of TestWikiLogicCRUD.test_get_logic ______________

    @pytest.fixture
    def tmp_wiki_path():
        """临时 wiki 目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            wiki_path = os.path.join(tmpdir, "test_wiki")
>           init_factor_wiki(wiki_path)

tests/research/test_wiki.py:21: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888b817b9d0>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
_________ ERROR at setup of TestWikiLogicCRUD.test_get_logic_not_found _________

    @pytest.fixture
    def tmp_wiki_path():
        """临时 wiki 目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            wiki_path = os.path.join(tmpdir, "test_wiki")
>           init_factor_wiki(wiki_path)

tests/research/test_wiki.py:21: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x78889a999650>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
____________ ERROR at setup of TestWikiLogicCRUD.test_search_logics ____________

    @pytest.fixture
    def tmp_wiki_path():
        """临时 wiki 目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            wiki_path = os.path.join(tmpdir, "test_wiki")
>           init_factor_wiki(wiki_path)

tests/research/test_wiki.py:21: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888b8b45150>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
______________ ERROR at setup of TestRelations.test_add_relation _______________

    @pytest.fixture
    def tmp_wiki_path():
        """临时 wiki 目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            wiki_path = os.path.join(tmpdir, "test_wiki")
>           init_factor_wiki(wiki_path)

tests/research/test_wiki.py:21: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x78893a6184d0>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
________ ERROR at setup of TestRelations.test_add_relation_invalid_type ________

    @pytest.fixture
    def tmp_wiki_path():
        """临时 wiki 目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            wiki_path = os.path.join(tmpdir, "test_wiki")
>           init_factor_wiki(wiki_path)

tests/research/test_wiki.py:21: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888a0060a90>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
______________ ERROR at setup of TestRelations.test_get_neighbors ______________

    @pytest.fixture
    def tmp_wiki_path():
        """临时 wiki 目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            wiki_path = os.path.join(tmpdir, "test_wiki")
>           init_factor_wiki(wiki_path)

tests/research/test_wiki.py:21: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888a009b090>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
______ ERROR at setup of TestICIRSerialization.test_icir_fields_roundtrip ______

    @pytest.fixture
    def tmp_wiki_path():
        """临时 wiki 目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            wiki_path = os.path.join(tmpdir, "test_wiki")
>           init_factor_wiki(wiki_path)

tests/research/test_wiki.py:21: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888b8b91110>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
_______ ERROR at setup of TestStrategyYaml.test_strategy_yaml_roundtrip ________

    @pytest.fixture
    def tmp_wiki_path():
        """临时 wiki 目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            wiki_path = os.path.join(tmpdir, "test_wiki")
>           init_factor_wiki(wiki_path)

tests/research/test_wiki.py:21: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888b9578c90>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
____________________ ERROR at setup of TestStatus.test_ping ____________________

    @pytest.fixture
    def tmp_wiki_path():
        """临时 wiki 目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            wiki_path = os.path.join(tmpdir, "test_wiki")
>           init_factor_wiki(wiki_path)

tests/research/test_wiki.py:21: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888b95b30d0>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
___________________ ERROR at setup of TestStatus.test_status ___________________

    @pytest.fixture
    def tmp_wiki_path():
        """临时 wiki 目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            wiki_path = os.path.join(tmpdir, "test_wiki")
>           init_factor_wiki(wiki_path)

tests/research/test_wiki.py:21: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888b8170bd0>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
__________ ERROR at setup of TestWikiStrategyCRUD.test_store_strategy __________

    @pytest.fixture
    def tmp_wiki_path():
        """临时 wiki 目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            wiki_path = os.path.join(tmpdir, "test_wiki")
>           init_factor_wiki(wiki_path)

tests/research/test_wiki.py:21: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888b81bd7d0>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
___________ ERROR at setup of TestWikiStrategyCRUD.test_get_strategy ___________

    @pytest.fixture
    def tmp_wiki_path():
        """临时 wiki 目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            wiki_path = os.path.join(tmpdir, "test_wiki")
>           init_factor_wiki(wiki_path)

tests/research/test_wiki.py:21: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888b8be0350>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
______ ERROR at setup of TestWikiStrategyCRUD.test_get_strategy_not_found ______

    @pytest.fixture
    def tmp_wiki_path():
        """临时 wiki 目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            wiki_path = os.path.join(tmpdir, "test_wiki")
>           init_factor_wiki(wiki_path)

tests/research/test_wiki.py:21: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888b81d3450>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
_________ ERROR at setup of TestWikiStrategyCRUD.test_list_strategies __________

    @pytest.fixture
    def tmp_wiki_path():
        """临时 wiki 目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            wiki_path = os.path.join(tmpdir, "test_wiki")
>           init_factor_wiki(wiki_path)

tests/research/test_wiki.py:21: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888b814c050>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
_ ERROR at setup of TestWikiStrategyCRUD.test_list_strategies_with_category_filter _

    @pytest.fixture
    def tmp_wiki_path():
        """临时 wiki 目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            wiki_path = os.path.join(tmpdir, "test_wiki")
>           init_factor_wiki(wiki_path)

tests/research/test_wiki.py:21: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888b81e1010>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
_ ERROR at setup of TestWikiStrategyCRUD.test_list_strategies_with_tags_filter _

    @pytest.fixture
    def tmp_wiki_path():
        """临时 wiki 目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            wiki_path = os.path.join(tmpdir, "test_wiki")
>           init_factor_wiki(wiki_path)

tests/research/test_wiki.py:21: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888b8b96290>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
_ ERROR at setup of TestWikiStrategyCRUD.test_strategy_roundtrip_with_yaml_and_json _

    @pytest.fixture
    def tmp_wiki_path():
        """临时 wiki 目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            wiki_path = os.path.join(tmpdir, "test_wiki")
>           init_factor_wiki(wiki_path)

tests/research/test_wiki.py:21: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888b8b92190>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
______ ERROR at setup of TestWikiReproductionCRUD.test_store_reproduction ______

    @pytest.fixture
    def tmp_wiki_path():
        """临时 wiki 目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            wiki_path = os.path.join(tmpdir, "test_wiki")
>           init_factor_wiki(wiki_path)

tests/research/test_wiki.py:21: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888a007bf10>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
_______ ERROR at setup of TestWikiReproductionCRUD.test_get_reproduction _______

    @pytest.fixture
    def tmp_wiki_path():
        """临时 wiki 目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            wiki_path = os.path.join(tmpdir, "test_wiki")
>           init_factor_wiki(wiki_path)

tests/research/test_wiki.py:21: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x78889a9ac7d0>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
__ ERROR at setup of TestWikiReproductionCRUD.test_get_reproduction_not_found __

    @pytest.fixture
    def tmp_wiki_path():
        """临时 wiki 目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            wiki_path = os.path.join(tmpdir, "test_wiki")
>           init_factor_wiki(wiki_path)

tests/research/test_wiki.py:21: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888ea37b250>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
_ ERROR at setup of TestWikiReproductionCRUD.test_store_reproduction_slash_in_title _

    @pytest.fixture
    def tmp_wiki_path():
        """临时 wiki 目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            wiki_path = os.path.join(tmpdir, "test_wiki")
>           init_factor_wiki(wiki_path)

tests/research/test_wiki.py:21: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888a0052350>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
=================================== FAILURES ===================================
_________ TestBuildDefaultClient.test_builds_client_with_valid_config __________

self = <test_llm_factory.TestBuildDefaultClient object at 0x78893a928cd0>
tmp_path = PosixPath('/tmp/pytest-of-ll/pytest-4160/test_builds_client_with_valid_0')
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7888d28ff490>

    def test_builds_client_with_valid_config(self, tmp_path: Path, monkeypatch) -> None:
        """有效 config 时 build_default_client 成功."""
        config_file = tmp_path / "llmwikify.json"
        config_file.write_text(
            json.dumps({
                "llm": {
                    "enabled": True,
                    "model": "minimax",
                    "api_key": "test-key",
                    "base_url": "https://api.test.com",
                }
            }),
            encoding="utf-8",
        )
        monkeypatch.setattr(lf, "CONFIG_PATH", config_file)
        # 可能因为 env 缺失抛错, 也可能成功
        try:
>           client = lf.build_default_client()
                     ^^^^^^^^^^^^^^^^^^^^^^^^^

tests/research/test_llm_factory.py:80: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/common/llm_factory.py:49: in build_default_client
    return build_llm_client(model=model)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
QuantNodes/research/common/llm/client.py:117: in build_llm_client
    from .streamable import StreamableLLMClient
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

    """Streamable LLM client — supports streaming, async, and function calling.
    
    Canonical location for the streaming-capable LLM client. The historical
    home in ``llmwikify.agent.backend.adapters`` is preserved as a thin
    deprecation shim; new code should import from
    ``llmwikify.foundation.llm.streamable`` instead.
    
    Usage::
    
        from QuantNodes.research.common.llm.streamable import StreamableLLMClient
    
        client = StreamableLLMClient.from_config(config_dict)
        text = client.chat(messages, temperature=0.3)
        async for chunk in client.astream_chat(messages):
            ...
    
    Token budget checking is applied automatically via decorator.
    Pass ``_prompt_name="..."`` in generation_params to label calls in logs.
    """
    
    from __future__ import annotations
    
    import json
    import logging
    import os
    import random
    import threading
    import time
    from dataclasses import dataclass, field
    from typing import Any
    
>   from ..llm_client import LLMClient, _legacy_fallback_enabled
E   ModuleNotFoundError: No module named 'QuantNodes.research.common.llm_client'

QuantNodes/research/common/llm/streamable.py:32: ModuleNotFoundError

During handling of the above exception, another exception occurred:

self = <test_llm_factory.TestBuildDefaultClient object at 0x78893a928cd0>
tmp_path = PosixPath('/tmp/pytest-of-ll/pytest-4160/test_builds_client_with_valid_0')
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7888d28ff490>

    def test_builds_client_with_valid_config(self, tmp_path: Path, monkeypatch) -> None:
        """有效 config 时 build_default_client 成功."""
        config_file = tmp_path / "llmwikify.json"
        config_file.write_text(
            json.dumps({
                "llm": {
                    "enabled": True,
                    "model": "minimax",
                    "api_key": "test-key",
                    "base_url": "https://api.test.com",
                }
            }),
            encoding="utf-8",
        )
        monkeypatch.setattr(lf, "CONFIG_PATH", config_file)
        # 可能因为 env 缺失抛错, 也可能成功
        try:
            client = lf.build_default_client()
            assert client is not None
        except Exception as exc:
            # 失败: 接受 (有 env 依赖)
            err_msg = str(exc).lower()
>           assert any(k in err_msg for k in ["config", "key", "enabled", "disabled"])
E           assert False
E            +  where False = any(<generator object TestBuildDefaultClient.test_builds_client_with_valid_config.<locals>.<genexpr> at 0x7888ea3765e0>)

tests/research/test_llm_factory.py:85: AssertionError
_ TestOrchestratorEnd2End.test_defer_section_detector_continues_with_no_sections _

self = <test_orchestrator.TestOrchestratorEnd2End object at 0x78893a9c90d0>
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7888d1f2a490>
tmp_path = PosixPath('/tmp/pytest-of-ll/pytest-4160/test_defer_section_detector_co0')

    def test_defer_section_detector_continues_with_no_sections(self, monkeypatch, tmp_path):
        """Stage 1 Call 1 DeferError → queue + sections=None."""
        from QuantNodes.research.paper_understanding.llm_extraction import orchestrator
    
    
        # Stage 0 stub
        def fake_stage0(source, output_root, paper_id, **_):
            work_dir = output_root / paper_id
            work_dir.mkdir(parents=True, exist_ok=True)
            (work_dir / "parsed.md").write_text("paper text " * 50, encoding="utf-8")
            return _StubStage0(text="paper text " * 50, paper_id=paper_id)
    
        monkeypatch.setattr(orchestrator, "run_stage0_ingest", fake_stage0)
    
        # detect_sections raises DeferError
        def fake_detect(*args, **kwargs):
            raise DeferError("section detect failed after 3 attempts")
    
        monkeypatch.setattr(orchestrator, "detect_sections", fake_detect)
    
        # planner returns valid plan (via _run_planner wrapper)
        def fake_plan(*args, **kwargs):
            return PlanResult(
                paper_id="p1", schema_choice="factor",
                n_signals_estimate=3, confidence=0.9,
                token_budget={"track_b_pass1": 5000}, success=True,
            )
    
        # track_a returns empty
        def fake_track_a(*args, **kwargs):
            return TrackAResult(
                paper_id="p1", schema_choice="factor", tier1={},
                success=True, latency_ms_total=100, llm_calls=1,
            )
    
        # track_b returns empty
        def fake_track_b(*args, **kwargs):
            return TrackBResult(
                paper_id="p1", schema_choice="factor", enabled=False,
                success=True, llm_calls=0,
            )
    
        monkeypatch.setattr(orchestrator, "_run_planner", fake_plan)
        monkeypatch.setattr(orchestrator, "run_track_a", fake_track_a)
        monkeypatch.setattr(orchestrator, "run_track_b", fake_track_b)
    
        # Run
>       result = run_one_paper(
            paper_id="test_paper",
            source_path=tmp_path / "fake.pdf",
            output_root=tmp_path / "papers",
        )

tests/research/test_orchestrator.py:391: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/paper_understanding/llm_extraction/orchestrator.py:256: in run_one_paper
    client = llm_client or build_default_client()
                           ^^^^^^^^^^^^^^^^^^^^^^
QuantNodes/research/common/llm_factory.py:49: in build_default_client
    return build_llm_client(model=model)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
QuantNodes/research/common/llm/client.py:117: in build_llm_client
    from .streamable import StreamableLLMClient
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

    """Streamable LLM client — supports streaming, async, and function calling.
    
    Canonical location for the streaming-capable LLM client. The historical
    home in ``llmwikify.agent.backend.adapters`` is preserved as a thin
    deprecation shim; new code should import from
    ``llmwikify.foundation.llm.streamable`` instead.
    
    Usage::
    
        from QuantNodes.research.common.llm.streamable import StreamableLLMClient
    
        client = StreamableLLMClient.from_config(config_dict)
        text = client.chat(messages, temperature=0.3)
        async for chunk in client.astream_chat(messages):
            ...
    
    Token budget checking is applied automatically via decorator.
    Pass ``_prompt_name="..."`` in generation_params to label calls in logs.
    """
    
    from __future__ import annotations
    
    import json
    import logging
    import os
    import random
    import threading
    import time
    from dataclasses import dataclass, field
    from typing import Any
    
>   from ..llm_client import LLMClient, _legacy_fallback_enabled
E   ModuleNotFoundError: No module named 'QuantNodes.research.common.llm_client'

QuantNodes/research/common/llm/streamable.py:32: ModuleNotFoundError
------------------------------ Captured log call -------------------------------
WARNING  QuantNodes.research.paper_understanding.llm_extraction.orchestrator:orchestrator.py:230 [orchestrator] paper=test_paper [2/5] stage1_call1: deferred: section detect failed after 3 attempts
________________ TestOrchestratorEnd2End.test_no_defer_no_queue ________________

self = <test_orchestrator.TestOrchestratorEnd2End object at 0x78893a9c97d0>
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x7888d0fc6610>
tmp_path = PosixPath('/tmp/pytest-of-ll/pytest-4160/test_no_defer_no_queue0')

    def test_no_defer_no_queue(self, monkeypatch, tmp_path):
        """Happy path: no DeferError → deferred_count=0."""
        from QuantNodes.research.paper_understanding.llm_extraction import orchestrator
    
    
        def fake_stage0(source, output_root, paper_id, **_):
            work_dir = output_root / paper_id
            work_dir.mkdir(parents=True, exist_ok=True)
            (work_dir / "parsed.md").write_text("text " * 50, encoding="utf-8")
            return _StubStage0(text="text " * 50, paper_id=paper_id)
    
        monkeypatch.setattr(orchestrator, "run_stage0_ingest", fake_stage0)
        monkeypatch.setattr(orchestrator, "detect_sections",
            lambda *a, **k: SectionDetectionResult(
                paper_id="p1", success=True, sections=[], latency_ms=0,
            ))
        monkeypatch.setattr(orchestrator, "_run_planner",
            lambda *a, **k: PlanResult(
                paper_id="p1", schema_choice="factor",
                n_signals_estimate=3, confidence=0.9,
                token_budget={"track_b_pass1": 5000}, success=True,
            ))
        monkeypatch.setattr(orchestrator, "run_track_a",
            lambda *a, **k: TrackAResult(
                paper_id="p1", schema_choice="factor", tier1={},
                success=True, latency_ms_total=100, llm_calls=1,
            ))
        monkeypatch.setattr(orchestrator, "run_track_b",
            lambda *a, **k: TrackBResult(
                paper_id="p1", schema_choice="factor", enabled=False,
                success=True, llm_calls=0,
            ))
    
>       result = run_one_paper(
            paper_id="p1",
            source_path=tmp_path / "fake.pdf",
            output_root=tmp_path / "papers",
        )

tests/research/test_orchestrator.py:508: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/paper_understanding/llm_extraction/orchestrator.py:256: in run_one_paper
    client = llm_client or build_default_client()
                           ^^^^^^^^^^^^^^^^^^^^^^
QuantNodes/research/common/llm_factory.py:49: in build_default_client
    return build_llm_client(model=model)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
QuantNodes/research/common/llm/client.py:117: in build_llm_client
    from .streamable import StreamableLLMClient
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

    """Streamable LLM client — supports streaming, async, and function calling.
    
    Canonical location for the streaming-capable LLM client. The historical
    home in ``llmwikify.agent.backend.adapters`` is preserved as a thin
    deprecation shim; new code should import from
    ``llmwikify.foundation.llm.streamable`` instead.
    
    Usage::
    
        from QuantNodes.research.common.llm.streamable import StreamableLLMClient
    
        client = StreamableLLMClient.from_config(config_dict)
        text = client.chat(messages, temperature=0.3)
        async for chunk in client.astream_chat(messages):
            ...
    
    Token budget checking is applied automatically via decorator.
    Pass ``_prompt_name="..."`` in generation_params to label calls in logs.
    """
    
    from __future__ import annotations
    
    import json
    import logging
    import os
    import random
    import threading
    import time
    from dataclasses import dataclass, field
    from typing import Any
    
>   from ..llm_client import LLMClient, _legacy_fallback_enabled
E   ModuleNotFoundError: No module named 'QuantNodes.research.common.llm_client'

QuantNodes/research/common/llm/streamable.py:32: ModuleNotFoundError
____________ TestTrackATier1Retry.test_retries_on_transient_failure ____________

self = <test_retry_integration.TestTrackATier1Retry object at 0x78893a72de50>

    def test_retries_on_transient_failure(self):
        good_response = json.dumps({
            "paper_metadata": {"title": "T", "authors": ["A"]},
        })
        client = FlakyClient(good_response, fail_times=2)
        plan = PlanResult(
            paper_id="p1", schema_choice="factor",
            n_signals_estimate=5, confidence=0.9,
            token_budget={"track_a_tier1": 5000}, success=True,
        )
>       result, latency = track_a._run_tier1(
            client, plan, "p1", "T", "x" * 200, None,
        )

tests/research/test_retry_integration.py:129: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/paper_understanding/llm_extraction/track_a.py:162: in _run_tier1
    system_text, user_template, params = _load_prompt(prompt_file)
                                         ^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

prompt_file = 'repro_extract_factor.yaml'

    def _load_prompt(prompt_file: str) -> tuple[str, str, dict[str, Any]]:
        path = PROMPTS_DIR / prompt_file
        if not path.exists():
>           raise FileNotFoundError(f"Prompt not found: {path}")
E           FileNotFoundError: Prompt not found: /home/ll/Public/QuantNodes/QuantNodes/foundation/prompts/_defaults/repro_extract_factor.yaml

QuantNodes/research/paper_understanding/llm_extraction/track_a.py:76: FileNotFoundError
____________ TestTrackATier2Retry.test_retries_on_transient_failure ____________

self = <test_retry_integration.TestTrackATier2Retry object at 0x78893a72e550>

    def test_retries_on_transient_failure(self):
        good_response = json.dumps({"backtest": "results"})
        client = FlakyClient(good_response, fail_times=2)
        plan = PlanResult(
            paper_id="p1", schema_choice="factor",
            n_signals_estimate=5, confidence=0.9,
            token_budget={"track_a_tier2_per_section": 5000}, success=True,
        )
        tier2, attempted, failed, latency = track_a._run_tier2(
            client, plan, "p1", "x" * 200,
        )
        # FlakyClient is cumulative: first 2 calls fail, rest succeed.
        # 5 tier2 prompts: prompt 1 takes 3 calls (2 fail + 1 ok),
        # prompts 2-5 take 1 call each. Total = 3 + 4 = 7.
>       assert client.calls == 7
E       assert 0 == 7
E        +  where 0 = <test_retry_integration.FlakyClient object at 0x7888e9f03e10>.calls

tests/research/test_retry_integration.py:154: AssertionError
------------------------------ Captured log call -------------------------------
WARNING  QuantNodes.research.paper_understanding.llm_extraction.track_a:track_a.py:259 [track_a] tier2 backtest paper=p1 error: Prompt not found: /home/ll/Public/QuantNodes/QuantNodes/foundation/prompts/_defaults/repro_extract_tier2_backtest.yaml
WARNING  QuantNodes.research.paper_understanding.llm_extraction.track_a:track_a.py:259 [track_a] tier2 performance paper=p1 error: Prompt not found: /home/ll/Public/QuantNodes/QuantNodes/foundation/prompts/_defaults/repro_extract_tier2_performance.yaml
WARNING  QuantNodes.research.paper_understanding.llm_extraction.track_a:track_a.py:259 [track_a] tier2 risk paper=p1 error: Prompt not found: /home/ll/Public/QuantNodes/QuantNodes/foundation/prompts/_defaults/repro_extract_tier2_risk.yaml
WARNING  QuantNodes.research.paper_understanding.llm_extraction.track_a:track_a.py:259 [track_a] tier2 implementation paper=p1 error: Prompt not found: /home/ll/Public/QuantNodes/QuantNodes/foundation/prompts/_defaults/repro_extract_tier2_implementation.yaml
WARNING  QuantNodes.research.paper_understanding.llm_extraction.track_a:track_a.py:259 [track_a] tier2 datasets paper=p1 error: Prompt not found: /home/ll/Public/QuantNodes/QuantNodes/foundation/prompts/_defaults/repro_extract_tier2_datasets.yaml
_____________ TestTrackATier2Retry.test_all_tier2_prompts_succeed ______________

self = <test_retry_integration.TestTrackATier2Retry object at 0x78893a72ead0>

    def test_all_tier2_prompts_succeed(self):
        """If retries succeed for all prompts, tier2 is fully populated."""
        good_response = json.dumps({"backtest": "results"})
        client = FlakyClient(good_response, fail_times=0)  # no failures
        plan = PlanResult(
            paper_id="p1", schema_choice="factor",
            n_signals_estimate=5, confidence=0.9,
            token_budget={"track_a_tier2_per_section": 5000}, success=True,
        )
        tier2, attempted, failed, latency = track_a._run_tier2(
            client, plan, "p1", "x" * 200,
        )
>       assert client.calls == 5  # 1 call per prompt, no retries
        ^^^^^^^^^^^^^^^^^^^^^^^^
E       assert 0 == 5
E        +  where 0 = <test_retry_integration.FlakyClient object at 0x7888e9f00e10>.calls

tests/research/test_retry_integration.py:171: AssertionError
------------------------------ Captured log call -------------------------------
WARNING  QuantNodes.research.paper_understanding.llm_extraction.track_a:track_a.py:259 [track_a] tier2 backtest paper=p1 error: Prompt not found: /home/ll/Public/QuantNodes/QuantNodes/foundation/prompts/_defaults/repro_extract_tier2_backtest.yaml
WARNING  QuantNodes.research.paper_understanding.llm_extraction.track_a:track_a.py:259 [track_a] tier2 performance paper=p1 error: Prompt not found: /home/ll/Public/QuantNodes/QuantNodes/foundation/prompts/_defaults/repro_extract_tier2_performance.yaml
WARNING  QuantNodes.research.paper_understanding.llm_extraction.track_a:track_a.py:259 [track_a] tier2 risk paper=p1 error: Prompt not found: /home/ll/Public/QuantNodes/QuantNodes/foundation/prompts/_defaults/repro_extract_tier2_risk.yaml
WARNING  QuantNodes.research.paper_understanding.llm_extraction.track_a:track_a.py:259 [track_a] tier2 implementation paper=p1 error: Prompt not found: /home/ll/Public/QuantNodes/QuantNodes/foundation/prompts/_defaults/repro_extract_tier2_implementation.yaml
WARNING  QuantNodes.research.paper_understanding.llm_extraction.track_a:track_a.py:259 [track_a] tier2 datasets paper=p1 error: Prompt not found: /home/ll/Public/QuantNodes/QuantNodes/foundation/prompts/_defaults/repro_extract_tier2_datasets.yaml
_______ TestTrackBPass1MultiTurnRetry.test_retries_on_transient_failure ________

self = <test_retry_integration.TestTrackBPass1MultiTurnRetry object at 0x78893a72f1d0>

    def test_retries_on_transient_failure(self):
        """LLM first call fails, second call succeeds."""
        good_response = json.dumps({
            "signals": [
                {"name": "S1", "formula": "rank(x)"},
                {"name": "S2", "formula": "rank(y)"},
            ],
            "done": True,
        })
        client = FlakyClient(good_response, fail_times=2)
        plan = PlanResult(
            paper_id="p1", schema_choice="factor",
            n_signals_estimate=2,
            confidence=0.9,
            token_budget={"track_b_pass1": PASS1_MAX_TOKENS_DEFAULT},
            success=True,
        )
>       stubs, latency, n_calls = track_b._run_pass1(
            client, plan, "p1", "x" * 200,
        )

tests/research/test_retry_integration.py:196: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/paper_understanding/llm_extraction/track_b.py:876: in _run_pass1
    system_text, user_template, params = _load_prompt(PROMPT_PASS1)
                                         ^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

prompt_file = 'repro_extract_track_b_pass1.yaml'

    def _load_prompt(prompt_file: str) -> tuple[str, str, dict[str, Any]]:
        path = (
            Path(__file__).parent.parent.parent.parent
            / "foundation" / "prompts" / "_defaults"
            / prompt_file
        )
        if not path.exists():
>           raise FileNotFoundError(f"Prompt not found: {path}")
E           FileNotFoundError: Prompt not found: /home/ll/Public/QuantNodes/QuantNodes/foundation/prompts/_defaults/repro_extract_track_b_pass1.yaml

QuantNodes/research/paper_understanding/llm_extraction/track_b.py:191: FileNotFoundError
____________ TestTrackBPass2Retry.test_retries_on_transient_failure ____________

self = <test_retry_integration.TestTrackBPass2Retry object at 0x78893a72f950>

    def test_retries_on_transient_failure(self):
        """Test that adaptive multi-turn can extract L1-L4 (success path).
    
        Note: Pass 2 prompt is now batch mode (v2). The retry mechanism
        for transient failures is tested separately in test_retry module.
        """
        good_response = json.dumps({
            "factors": [
                {
                    "name": "S1",
                    "description": "desc",
                    "l1": {"formula": "x+y"},
                    "l2": {"function_calls": ["rank"]},
                    "l3": {"input_data": ["close"]},
                    "l4": {"strategy_type": "mean-reversion"},
                    "need_more_context": None,
                }
            ]
        })
        # No failures - test success path
        client = FlakyClient(good_response, fail_times=0)
        plan = PlanResult(
            paper_id="p1", schema_choice="factor",
            n_signals_estimate=1, confidence=0.9,
            token_budget={"track_b_pass2_per_factor": 5000}, success=True,
        )
        stub = SignalStub(
            index=1, name="S1", formula_brief="x+y",
            context_excerpt="x" * 1000,  # > 50 chars to avoid fallback
        )
        # Test via _run_pass2_adaptive (async, single signal)
        import asyncio
>       details, latency = asyncio.run(
            track_b._run_pass2_adaptive(
                client, plan, "p1", [stub], "x" * 200,
            )
        )

tests/research/test_retry_integration.py:241: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
/usr/lib/python3.11/asyncio/runners.py:190: in run
    return runner.run(main)
           ^^^^^^^^^^^^^^^^
/usr/lib/python3.11/asyncio/runners.py:118: in run
    return self._loop.run_until_complete(task)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/usr/lib/python3.11/asyncio/base_events.py:654: in run_until_complete
    return future.result()
           ^^^^^^^^^^^^^^^
QuantNodes/research/paper_understanding/llm_extraction/track_b.py:1380: in _run_pass2_adaptive
    system_text, user_template, params = _load_prompt(prompt_file)
                                         ^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

prompt_file = 'repro_extract_track_b_pass2.yaml'

    def _load_prompt(prompt_file: str) -> tuple[str, str, dict[str, Any]]:
        path = (
            Path(__file__).parent.parent.parent.parent
            / "foundation" / "prompts" / "_defaults"
            / prompt_file
        )
        if not path.exists():
>           raise FileNotFoundError(f"Prompt not found: {path}")
E           FileNotFoundError: Prompt not found: /home/ll/Public/QuantNodes/QuantNodes/foundation/prompts/_defaults/repro_extract_track_b_pass2.yaml

QuantNodes/research/paper_understanding/llm_extraction/track_b.py:191: FileNotFoundError
______ TestAdaptiveMultiTurn.test_completes_batch_with_sufficient_context ______

self = <test_track_b_adaptive_helpers.TestAdaptiveMultiTurn object at 0x78893a60fc50>

    def test_completes_batch_with_sufficient_context(self):
        """All signals complete in 1 round (no need_more_context)."""
        client = MagicMock()
        client.achat = AsyncMock(return_value=json.dumps({
            "factors": [
                {
                    "name": "Alpha#1", "description": "d1",
                    "l1": {"definition": "def1", "formula": "f1(x)"},
                    "l2": {}, "l3": {}, "l4": {"hypotheses": []},
                    "need_more_context": None,
                },
                {
                    "name": "Alpha#2", "description": "d2",
                    "l1": {"definition": "def2", "formula": "f2(x)"},
                    "l2": {}, "l3": {}, "l4": {"hypotheses": []},
                    "need_more_context": None,
                },
                {
                    "name": "Alpha#3", "description": "d3",
                    "l1": {"definition": "def3", "formula": "f3(x)"},
                    "l2": {}, "l3": {}, "l4": {"hypotheses": []},
                    "need_more_context": None,
                },
            ]
        }))
        stubs = self._make_stubs(3)
>       details, latency = asyncio.run(
            _run_pass2_adaptive(
                client, self._make_plan(), "test", stubs, "x" * 1000,
            )
        )

tests/research/test_track_b_adaptive_helpers.py:264: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
/usr/lib/python3.11/asyncio/runners.py:190: in run
    return runner.run(main)
           ^^^^^^^^^^^^^^^^
/usr/lib/python3.11/asyncio/runners.py:118: in run
    return self._loop.run_until_complete(task)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/usr/lib/python3.11/asyncio/base_events.py:654: in run_until_complete
    return future.result()
           ^^^^^^^^^^^^^^^
QuantNodes/research/paper_understanding/llm_extraction/track_b.py:1380: in _run_pass2_adaptive
    system_text, user_template, params = _load_prompt(prompt_file)
                                         ^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

prompt_file = 'repro_extract_track_b_pass2.yaml'

    def _load_prompt(prompt_file: str) -> tuple[str, str, dict[str, Any]]:
        path = (
            Path(__file__).parent.parent.parent.parent
            / "foundation" / "prompts" / "_defaults"
            / prompt_file
        )
        if not path.exists():
>           raise FileNotFoundError(f"Prompt not found: {path}")
E           FileNotFoundError: Prompt not found: /home/ll/Public/QuantNodes/QuantNodes/foundation/prompts/_defaults/repro_extract_track_b_pass2.yaml

QuantNodes/research/paper_understanding/llm_extraction/track_b.py:191: FileNotFoundError
_____ TestAdaptiveMultiTurn.test_handles_need_more_context_with_supplement _____

self = <test_track_b_adaptive_helpers.TestAdaptiveMultiTurn object at 0x78893a618250>

    def test_handles_need_more_context_with_supplement(self):
        """Signal that needs more context gets a supplement, then completes."""
        # Round 1: signal 1 needs more, others complete
        # Round 2 (after supplement): signal 1 completes
        client = MagicMock()
        client.achat = AsyncMock(side_effect=[
            json.dumps({
                "factors": [
                    {
                        "name": "Alpha#1", "description": "needs more",
                        "l1": None, "l2": None, "l3": None, "l4": None,
                        "need_more_context": {
                            "level": "a", "reason": "missing params",
                        },
                    },
                    {
                        "name": "Alpha#2", "description": "d2",
                        "l1": {"definition": "def2"}, "l2": {}, "l3": {}, "l4": {},
                        "need_more_context": None,
                    },
                    {
                        "name": "Alpha#3", "description": "d3",
                        "l1": {"definition": "def3"}, "l2": {}, "l3": {}, "l4": {},
                        "need_more_context": None,
                    },
                ]
            }),
            # Round 2: after supplement, Alpha#1 completes
            json.dumps({
                "factors": [
                    {
                        "name": "Alpha#1", "description": "d1 now complete",
                        "l1": {"definition": "def1 with params"},
                        "l2": {}, "l3": {}, "l4": {},
                        "need_more_context": None,
                    },
                ]
            }),
        ])
        stubs = self._make_stubs(3)
>       details, latency = asyncio.run(
            _run_pass2_adaptive(
                client, self._make_plan(), "test", stubs, "x" * 1000,
            )
        )

tests/research/test_track_b_adaptive_helpers.py:313: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
/usr/lib/python3.11/asyncio/runners.py:190: in run
    return runner.run(main)
           ^^^^^^^^^^^^^^^^
/usr/lib/python3.11/asyncio/runners.py:118: in run
    return self._loop.run_until_complete(task)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/usr/lib/python3.11/asyncio/base_events.py:654: in run_until_complete
    return future.result()
           ^^^^^^^^^^^^^^^
QuantNodes/research/paper_understanding/llm_extraction/track_b.py:1380: in _run_pass2_adaptive
    system_text, user_template, params = _load_prompt(prompt_file)
                                         ^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

prompt_file = 'repro_extract_track_b_pass2.yaml'

    def _load_prompt(prompt_file: str) -> tuple[str, str, dict[str, Any]]:
        path = (
            Path(__file__).parent.parent.parent.parent
            / "foundation" / "prompts" / "_defaults"
            / prompt_file
        )
        if not path.exists():
>           raise FileNotFoundError(f"Prompt not found: {path}")
E           FileNotFoundError: Prompt not found: /home/ll/Public/QuantNodes/QuantNodes/foundation/prompts/_defaults/repro_extract_track_b_pass2.yaml

QuantNodes/research/paper_understanding/llm_extraction/track_b.py:191: FileNotFoundError
_______ TestAdaptiveMultiTurn.test_max_supplements_exceeded_marks_failed _______

self = <test_track_b_adaptive_helpers.TestAdaptiveMultiTurn object at 0x78893a618810>

    def test_max_supplements_exceeded_marks_failed(self):
        """After 5 supplements without success, mark as failed."""
        client = MagicMock()
        # All responses say need_more_context
        always_need_more = json.dumps({
            "factors": [
                {
                    "name": "Alpha#1",
                    "l1": None, "l2": None, "l3": None, "l4": None,
                    "need_more_context": {"level": "a", "reason": "still missing"},
                },
            ]
        })
        client.achat = AsyncMock(return_value=always_need_more)
        stubs = [self._make_stubs(1)[0]]  # 1 signal
>       details, latency = asyncio.run(
            _run_pass2_adaptive(
                client, self._make_plan(), "test", stubs, "x" * 1000,
            )
        )

tests/research/test_track_b_adaptive_helpers.py:337: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
/usr/lib/python3.11/asyncio/runners.py:190: in run
    return runner.run(main)
           ^^^^^^^^^^^^^^^^
/usr/lib/python3.11/asyncio/runners.py:118: in run
    return self._loop.run_until_complete(task)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/usr/lib/python3.11/asyncio/base_events.py:654: in run_until_complete
    return future.result()
           ^^^^^^^^^^^^^^^
QuantNodes/research/paper_understanding/llm_extraction/track_b.py:1380: in _run_pass2_adaptive
    system_text, user_template, params = _load_prompt(prompt_file)
                                         ^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

prompt_file = 'repro_extract_track_b_pass2.yaml'

    def _load_prompt(prompt_file: str) -> tuple[str, str, dict[str, Any]]:
        path = (
            Path(__file__).parent.parent.parent.parent
            / "foundation" / "prompts" / "_defaults"
            / prompt_file
        )
        if not path.exists():
>           raise FileNotFoundError(f"Prompt not found: {path}")
E           FileNotFoundError: Prompt not found: /home/ll/Public/QuantNodes/QuantNodes/foundation/prompts/_defaults/repro_extract_track_b_pass2.yaml

QuantNodes/research/paper_understanding/llm_extraction/track_b.py:191: FileNotFoundError
____ TestAdaptiveMultiTurn.test_json_parse_failure_continues_to_next_round _____

self = <test_track_b_adaptive_helpers.TestAdaptiveMultiTurn object at 0x78893a618e10>

    def test_json_parse_failure_continues_to_next_round(self):
        """If LLM returns unparseable JSON, continue trying."""
        client = MagicMock()
        client.achat = AsyncMock(side_effect=[
            "not valid json",  # Round 1: parse fail
            json.dumps({
                "factors": [
                    {
                        "name": "Alpha#1", "description": "d1",
                        "l1": {"definition": "def1"}, "l2": {}, "l3": {}, "l4": {},
                        "need_more_context": None,
                    },
                ]
            }),
        ])
        stubs = self._make_stubs(1)
>       details, latency = asyncio.run(
            _run_pass2_adaptive(
                client, self._make_plan(), "test", stubs, "x" * 1000,
            )
        )

tests/research/test_track_b_adaptive_helpers.py:363: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
/usr/lib/python3.11/asyncio/runners.py:190: in run
    return runner.run(main)
           ^^^^^^^^^^^^^^^^
/usr/lib/python3.11/asyncio/runners.py:118: in run
    return self._loop.run_until_complete(task)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/usr/lib/python3.11/asyncio/base_events.py:654: in run_until_complete
    return future.result()
           ^^^^^^^^^^^^^^^
QuantNodes/research/paper_understanding/llm_extraction/track_b.py:1380: in _run_pass2_adaptive
    system_text, user_template, params = _load_prompt(prompt_file)
                                         ^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

prompt_file = 'repro_extract_track_b_pass2.yaml'

    def _load_prompt(prompt_file: str) -> tuple[str, str, dict[str, Any]]:
        path = (
            Path(__file__).parent.parent.parent.parent
            / "foundation" / "prompts" / "_defaults"
            / prompt_file
        )
        if not path.exists():
>           raise FileNotFoundError(f"Prompt not found: {path}")
E           FileNotFoundError: Prompt not found: /home/ll/Public/QuantNodes/QuantNodes/foundation/prompts/_defaults/repro_extract_track_b_pass2.yaml

QuantNodes/research/paper_understanding/llm_extraction/track_b.py:191: FileNotFoundError
_______ TestAdaptiveMultiTurn.test_legacy_single_factor_format_supported _______

self = <test_track_b_adaptive_helpers.TestAdaptiveMultiTurn object at 0x78893a6193d0>

    def test_legacy_single_factor_format_supported(self):
        """Legacy `{"factor": {...}}` format still works."""
        client = MagicMock()
        client.achat = AsyncMock(return_value=json.dumps({
            "factor": {
                "name": "Alpha#1", "description": "d1",
                "l1": {"definition": "def1"}, "l2": {}, "l3": {}, "l4": {},
            }
        }))
        stubs = self._make_stubs(1)
>       details, latency = asyncio.run(
            _run_pass2_adaptive(
                client, self._make_plan(), "test", stubs, "x" * 1000,
            )
        )

tests/research/test_track_b_adaptive_helpers.py:382: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
/usr/lib/python3.11/asyncio/runners.py:190: in run
    return runner.run(main)
           ^^^^^^^^^^^^^^^^
/usr/lib/python3.11/asyncio/runners.py:118: in run
    return self._loop.run_until_complete(task)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/usr/lib/python3.11/asyncio/base_events.py:654: in run_until_complete
    return future.result()
           ^^^^^^^^^^^^^^^
QuantNodes/research/paper_understanding/llm_extraction/track_b.py:1380: in _run_pass2_adaptive
    system_text, user_template, params = _load_prompt(prompt_file)
                                         ^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

prompt_file = 'repro_extract_track_b_pass2.yaml'

    def _load_prompt(prompt_file: str) -> tuple[str, str, dict[str, Any]]:
        path = (
            Path(__file__).parent.parent.parent.parent
            / "foundation" / "prompts" / "_defaults"
            / prompt_file
        )
        if not path.exists():
>           raise FileNotFoundError(f"Prompt not found: {path}")
E           FileNotFoundError: Prompt not found: /home/ll/Public/QuantNodes/QuantNodes/foundation/prompts/_defaults/repro_extract_track_b_pass2.yaml

QuantNodes/research/paper_understanding/llm_extraction/track_b.py:191: FileNotFoundError
___________ TestAdaptiveMultiTurn.test_resume_skips_existing_details ___________

self = <test_track_b_adaptive_helpers.TestAdaptiveMultiTurn object at 0x78893a619a50>

    def test_resume_skips_existing_details(self):
        """Resume mode: skip signals already in existing_details."""
        client = MagicMock()
        client.achat = AsyncMock(return_value=json.dumps({
            "factors": [
                {
                    "name": "Alpha#3", "description": "d3",
                    "l1": {"definition": "def3"}, "l2": {}, "l3": {}, "l4": {},
                    "need_more_context": None,
                },
            ]
        }))
        existing = [
            SignalDetail(
                name="Alpha#1", description="d1",
                l1={"definition": "def1"}, success=True,
            ),
            SignalDetail(
                name="Alpha#2", description="d2",
                l1={"definition": "def2"}, success=True,
            ),
        ]
        stubs = self._make_stubs(3)
>       details, latency = asyncio.run(
            _run_pass2_adaptive(
                client, self._make_plan(), "test", stubs, "x" * 1000,
                existing_details=existing,
            )
        )

tests/research/test_track_b_adaptive_helpers.py:414: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
/usr/lib/python3.11/asyncio/runners.py:190: in run
    return runner.run(main)
           ^^^^^^^^^^^^^^^^
/usr/lib/python3.11/asyncio/runners.py:118: in run
    return self._loop.run_until_complete(task)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/usr/lib/python3.11/asyncio/base_events.py:654: in run_until_complete
    return future.result()
           ^^^^^^^^^^^^^^^
QuantNodes/research/paper_understanding/llm_extraction/track_b.py:1380: in _run_pass2_adaptive
    system_text, user_template, params = _load_prompt(prompt_file)
                                         ^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

prompt_file = 'repro_extract_track_b_pass2.yaml'

    def _load_prompt(prompt_file: str) -> tuple[str, str, dict[str, Any]]:
        path = (
            Path(__file__).parent.parent.parent.parent
            / "foundation" / "prompts" / "_defaults"
            / prompt_file
        )
        if not path.exists():
>           raise FileNotFoundError(f"Prompt not found: {path}")
E           FileNotFoundError: Prompt not found: /home/ll/Public/QuantNodes/QuantNodes/foundation/prompts/_defaults/repro_extract_track_b_pass2.yaml

QuantNodes/research/paper_understanding/llm_extraction/track_b.py:191: FileNotFoundError
________ TestAdaptiveMultiTurn.test_history_trimmed_after_max_messages _________

self = <test_track_b_adaptive_helpers.TestAdaptiveMultiTurn object at 0x78893a61a110>

    def test_history_trimmed_after_max_messages(self):
        """Test that messages list is trimmed when exceeding max_history_messages.
    
        This is an indirect test - we run 25 rounds of need_more_context
        (max 5 supplements per signal) and verify it doesn't crash from
        message accumulation.
        """
        client = MagicMock()
        # Always return need_more_context to force many rounds
        always_need_more = json.dumps({
            "factors": [
                {
                    "name": "Alpha#1",
                    "l1": None, "l2": None, "l3": None, "l4": None,
                    "need_more_context": {"level": "a", "reason": "need more"},
                },
            ]
        })
        client.achat = AsyncMock(return_value=always_need_more)
        stubs = [self._make_stubs(1)[0]]
        # Should not raise; max_supplements=5 will eventually mark failed
>       details, latency = asyncio.run(
            _run_pass2_adaptive(
                client, self._make_plan(), "test", stubs, "x" * 1000,
            )
        )

tests/research/test_track_b_adaptive_helpers.py:445: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
/usr/lib/python3.11/asyncio/runners.py:190: in run
    return runner.run(main)
           ^^^^^^^^^^^^^^^^
/usr/lib/python3.11/asyncio/runners.py:118: in run
    return self._loop.run_until_complete(task)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/usr/lib/python3.11/asyncio/base_events.py:654: in run_until_complete
    return future.result()
           ^^^^^^^^^^^^^^^
QuantNodes/research/paper_understanding/llm_extraction/track_b.py:1380: in _run_pass2_adaptive
    system_text, user_template, params = _load_prompt(prompt_file)
                                         ^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

prompt_file = 'repro_extract_track_b_pass2.yaml'

    def _load_prompt(prompt_file: str) -> tuple[str, str, dict[str, Any]]:
        path = (
            Path(__file__).parent.parent.parent.parent
            / "foundation" / "prompts" / "_defaults"
            / prompt_file
        )
        if not path.exists():
>           raise FileNotFoundError(f"Prompt not found: {path}")
E           FileNotFoundError: Prompt not found: /home/ll/Public/QuantNodes/QuantNodes/foundation/prompts/_defaults/repro_extract_track_b_pass2.yaml

QuantNodes/research/paper_understanding/llm_extraction/track_b.py:191: FileNotFoundError
_________ TestHybridPass2Orchestration.test_hybrid_phase1_phase2_merge _________

self = <test_track_b_hybrid.TestHybridPass2Orchestration object at 0x78893a64f350>
tmp_path = PosixPath('/tmp/pytest-of-ll/pytest-4160/test_hybrid_phase1_phase2_merg0')

    def test_hybrid_phase1_phase2_merge(self, tmp_path: Path):
        """Hybrid: parallel phase + adaptive phase + merge."""
        def response_factory(user_msg):
            # Extract Alpha#N from user message
            import re
            m = re.search(r"Alpha#(\d+)", user_msg)
            idx = int(m.group(1)) if m else 0
            # First 2 are shallow (need supplement)
            if idx < 2:
                return json.dumps({
                    "factors": [{
                        "name": f"Alpha#{idx}",
                        "description": "shallow",
                        "l1": {"formula": "rank(x)"},
                        "l3": {"intuition": "short", "theoretical_basis": "x",
                               "market_behavior": "y"},
                        "l4": {"hypotheses": ["h1"]},
                    }]
                })
            else:
                return json.dumps({
                    "factors": [{
                        "name": f"Alpha#{idx}",
                        "description": "deep",
                        "l1": {"formula": "rank(x)"},
                        "l3": {"intuition": "x" * 200, "theoretical_basis": "y" * 80,
                               "market_behavior": "normal"},
                        "l4": {"hypotheses": ["h1", "h2", "h3"]},
                    }]
                })
    
        client = self._make_async_mock_client(response_factory)
        plan = MagicMock()
        plan.schema_choice = "factor"
    
        stubs = [_stub(f"Alpha#{i}", index=i) for i in range(5)]
>       details, latency = _hybrid_pass2(
            client, plan, "test_paper", stubs, "paper text",
            work_dir=tmp_path,
        )

tests/research/test_track_b_hybrid.py:385: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/paper_understanding/llm_extraction/track_b.py:634: in _hybrid_pass2
    parallel_details, parallel_latency = asyncio.run(
/usr/lib/python3.11/asyncio/runners.py:190: in run
    return runner.run(main)
           ^^^^^^^^^^^^^^^^
/usr/lib/python3.11/asyncio/runners.py:118: in run
    return self._loop.run_until_complete(task)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/usr/lib/python3.11/asyncio/base_events.py:654: in run_until_complete
    return future.result()
           ^^^^^^^^^^^^^^^
QuantNodes/research/paper_understanding/llm_extraction/track_b.py:1627: in _run_pass2_parallel
    stub, detail = await coro
                   ^^^^^^^^^^
/usr/lib/python3.11/asyncio/tasks.py:615: in _wait_for_one
    return f.result()  # May raise f.exception().
           ^^^^^^^^^^
QuantNodes/research/paper_understanding/llm_extraction/track_b.py:1094: in _run_pass2_one_async
    system_text, user_template, params = _load_prompt(PROMPT_PASS2)
                                         ^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

prompt_file = 'repro_extract_track_b_pass2.yaml'

    def _load_prompt(prompt_file: str) -> tuple[str, str, dict[str, Any]]:
        path = (
            Path(__file__).parent.parent.parent.parent
            / "foundation" / "prompts" / "_defaults"
            / prompt_file
        )
        if not path.exists():
>           raise FileNotFoundError(f"Prompt not found: {path}")
E           FileNotFoundError: Prompt not found: /home/ll/Public/QuantNodes/QuantNodes/foundation/prompts/_defaults/repro_extract_track_b_pass2.yaml

QuantNodes/research/paper_understanding/llm_extraction/track_b.py:191: FileNotFoundError
______ TestHybridPass2Orchestration.test_hybrid_no_shallow_skips_adaptive ______

self = <test_track_b_hybrid.TestHybridPass2Orchestration object at 0x78893a64f950>
tmp_path = PosixPath('/tmp/pytest-of-ll/pytest-4160/test_hybrid_no_shallow_skips_a0')

    def test_hybrid_no_shallow_skips_adaptive(self, tmp_path: Path):
        """When all factors deep, hybrid skips adaptive phase."""
        call_log = []
    
        def response_factory(user_msg):
            call_log.append(user_msg)
            import re
            m = re.search(r"Alpha#(\d+)", user_msg)
            idx = int(m.group(1)) if m else 0
            return json.dumps({
                "factors": [{
                    "name": f"Alpha#{idx}",
                    "description": "deep",
                    "l1": {"formula": "rank(x)"},
                    "l3": {"intuition": "x" * 200, "theoretical_basis": "y" * 80,
                           "market_behavior": "normal"},
                    "l4": {"hypotheses": ["h1", "h2", "h3"]},
                }]
            })
    
        client = self._make_async_mock_client(response_factory)
        plan = MagicMock()
        plan.schema_choice = "factor"
    
        stubs = [_stub(f"Alpha#{i}", index=i) for i in range(3)]
>       details, latency = _hybrid_pass2(
            client, plan, "test_paper", stubs, "paper text",
            work_dir=tmp_path,
        )

tests/research/test_track_b_hybrid.py:417: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/paper_understanding/llm_extraction/track_b.py:634: in _hybrid_pass2
    parallel_details, parallel_latency = asyncio.run(
/usr/lib/python3.11/asyncio/runners.py:190: in run
    return runner.run(main)
           ^^^^^^^^^^^^^^^^
/usr/lib/python3.11/asyncio/runners.py:118: in run
    return self._loop.run_until_complete(task)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/usr/lib/python3.11/asyncio/base_events.py:654: in run_until_complete
    return future.result()
           ^^^^^^^^^^^^^^^
QuantNodes/research/paper_understanding/llm_extraction/track_b.py:1627: in _run_pass2_parallel
    stub, detail = await coro
                   ^^^^^^^^^^
/usr/lib/python3.11/asyncio/tasks.py:615: in _wait_for_one
    return f.result()  # May raise f.exception().
           ^^^^^^^^^^
QuantNodes/research/paper_understanding/llm_extraction/track_b.py:1094: in _run_pass2_one_async
    system_text, user_template, params = _load_prompt(PROMPT_PASS2)
                                         ^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

prompt_file = 'repro_extract_track_b_pass2.yaml'

    def _load_prompt(prompt_file: str) -> tuple[str, str, dict[str, Any]]:
        path = (
            Path(__file__).parent.parent.parent.parent
            / "foundation" / "prompts" / "_defaults"
            / prompt_file
        )
        if not path.exists():
>           raise FileNotFoundError(f"Prompt not found: {path}")
E           FileNotFoundError: Prompt not found: /home/ll/Public/QuantNodes/QuantNodes/foundation/prompts/_defaults/repro_extract_track_b_pass2.yaml

QuantNodes/research/paper_understanding/llm_extraction/track_b.py:191: FileNotFoundError
______________ TestSupplementPrompt.test_supplement_prompt_loads _______________

self = <test_track_b_hybrid.TestSupplementPrompt object at 0x78893a659390>

    def test_supplement_prompt_loads(self):
        """PROMPT_PASS2_SUPPLEMENT should load successfully."""
>       system, user, params = _load_prompt(PROMPT_PASS2_SUPPLEMENT)
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests/research/test_track_b_hybrid.py:456: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

prompt_file = 'repro_extract_track_b_pass2_supplement.yaml'

    def _load_prompt(prompt_file: str) -> tuple[str, str, dict[str, Any]]:
        path = (
            Path(__file__).parent.parent.parent.parent
            / "foundation" / "prompts" / "_defaults"
            / prompt_file
        )
        if not path.exists():
>           raise FileNotFoundError(f"Prompt not found: {path}")
E           FileNotFoundError: Prompt not found: /home/ll/Public/QuantNodes/QuantNodes/foundation/prompts/_defaults/repro_extract_track_b_pass2_supplement.yaml

QuantNodes/research/paper_understanding/llm_extraction/track_b.py:191: FileNotFoundError
__________ TestSupplementPrompt.test_supplement_prompt_requires_depth __________

self = <test_track_b_hybrid.TestSupplementPrompt object at 0x78893a659a10>

    def test_supplement_prompt_requires_depth(self):
        """Supplement prompt should mention depth requirements."""
>       system, user, params = _load_prompt(PROMPT_PASS2_SUPPLEMENT)
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests/research/test_track_b_hybrid.py:465: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

prompt_file = 'repro_extract_track_b_pass2_supplement.yaml'

    def _load_prompt(prompt_file: str) -> tuple[str, str, dict[str, Any]]:
        path = (
            Path(__file__).parent.parent.parent.parent
            / "foundation" / "prompts" / "_defaults"
            / prompt_file
        )
        if not path.exists():
>           raise FileNotFoundError(f"Prompt not found: {path}")
E           FileNotFoundError: Prompt not found: /home/ll/Public/QuantNodes/QuantNodes/foundation/prompts/_defaults/repro_extract_track_b_pass2_supplement.yaml

QuantNodes/research/paper_understanding/llm_extraction/track_b.py:191: FileNotFoundError
______ TestSupplementPrompt.test_supplement_prompt_differs_from_standard _______

self = <test_track_b_hybrid.TestSupplementPrompt object at 0x78893a65a110>

    def test_supplement_prompt_differs_from_standard(self):
        """Supplement prompt should be different from standard Pass 2."""
>       sys_sup, user_sup, _ = _load_prompt(PROMPT_PASS2_SUPPLEMENT)
                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

tests/research/test_track_b_hybrid.py:478: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

prompt_file = 'repro_extract_track_b_pass2_supplement.yaml'

    def _load_prompt(prompt_file: str) -> tuple[str, str, dict[str, Any]]:
        path = (
            Path(__file__).parent.parent.parent.parent
            / "foundation" / "prompts" / "_defaults"
            / prompt_file
        )
        if not path.exists():
>           raise FileNotFoundError(f"Prompt not found: {path}")
E           FileNotFoundError: Prompt not found: /home/ll/Public/QuantNodes/QuantNodes/foundation/prompts/_defaults/repro_extract_track_b_pass2_supplement.yaml

QuantNodes/research/paper_understanding/llm_extraction/track_b.py:191: FileNotFoundError
_____________________ TestRunPass1.test_one_round_all_done _____________________

self = <test_track_b_multiturn.TestRunPass1 object at 0x78893a669fd0>

    def test_one_round_all_done(self):
        """All 5 signals in first round, LLM says done: true → done after 1."""
        plan = PlanResult(
            paper_id="test",
            schema_choice="factor",
            n_signals_estimate=5,
            confidence=0.95,
            token_budget={"track_b_pass1": PASS1_MAX_TOKENS_DEFAULT},
            success=True,
        )
        client = RoundSequenceFakeLLM([
            single_round_response(5, done=True),
        ])
>       stubs, latency, n_calls = _run_pass1(
            client, plan, "test", "dummy text",
        )

tests/research/test_track_b_multiturn.py:132: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/paper_understanding/llm_extraction/track_b.py:876: in _run_pass1
    system_text, user_template, params = _load_prompt(PROMPT_PASS1)
                                         ^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

prompt_file = 'repro_extract_track_b_pass1.yaml'

    def _load_prompt(prompt_file: str) -> tuple[str, str, dict[str, Any]]:
        path = (
            Path(__file__).parent.parent.parent.parent
            / "foundation" / "prompts" / "_defaults"
            / prompt_file
        )
        if not path.exists():
>           raise FileNotFoundError(f"Prompt not found: {path}")
E           FileNotFoundError: Prompt not found: /home/ll/Public/QuantNodes/QuantNodes/foundation/prompts/_defaults/repro_extract_track_b_pass1.yaml

QuantNodes/research/paper_understanding/llm_extraction/track_b.py:191: FileNotFoundError
__________________ TestRunPass1.test_two_rounds_continuation ___________________

self = <test_track_b_multiturn.TestRunPass1 object at 0x78893a66a550>

    def test_two_rounds_continuation(self):
        """30 signals split into two rounds 20 + 10 → done by count."""
        plan = PlanResult(
            paper_id="test",
            schema_choice="factor",
            n_signals_estimate=30,
            confidence=0.95,
            token_budget={"track_b_pass1": PASS1_MAX_TOKENS_DEFAULT},
            success=True,
        )
        client = RoundSequenceFakeLLM([
            single_round_response(20, done=False, start=1),
            single_round_response(10, done=True, start=21),
        ])
>       stubs, latency, n_calls = _run_pass1(
            client, plan, "test", "dummy text",
        )

tests/research/test_track_b_multiturn.py:154: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/paper_understanding/llm_extraction/track_b.py:876: in _run_pass1
    system_text, user_template, params = _load_prompt(PROMPT_PASS1)
                                         ^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

prompt_file = 'repro_extract_track_b_pass1.yaml'

    def _load_prompt(prompt_file: str) -> tuple[str, str, dict[str, Any]]:
        path = (
            Path(__file__).parent.parent.parent.parent
            / "foundation" / "prompts" / "_defaults"
            / prompt_file
        )
        if not path.exists():
>           raise FileNotFoundError(f"Prompt not found: {path}")
E           FileNotFoundError: Prompt not found: /home/ll/Public/QuantNodes/QuantNodes/foundation/prompts/_defaults/repro_extract_track_b_pass1.yaml

QuantNodes/research/paper_understanding/llm_extraction/track_b.py:191: FileNotFoundError
__________________ TestRunPass1.test_done_by_llm_before_count __________________

self = <test_track_b_multiturn.TestRunPass1 object at 0x78893a66aad0>

    def test_done_by_llm_before_count(self):
        """LLM says done after 20 signals even though estimate is 30 → stop."""
        plan = PlanResult(
            paper_id="test",
            schema_choice="factor",
            n_signals_estimate=30,
            confidence=0.95,
            token_budget={"track_b_pass1": PASS1_MAX_TOKENS_DEFAULT},
            success=True,
        )
        client = RoundSequenceFakeLLM([
            single_round_response(20, done=True),
        ])
>       stubs, latency, n_calls = _run_pass1(
            client, plan, "test", "dummy text",
        )

tests/research/test_track_b_multiturn.py:178: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/paper_understanding/llm_extraction/track_b.py:876: in _run_pass1
    system_text, user_template, params = _load_prompt(PROMPT_PASS1)
                                         ^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

prompt_file = 'repro_extract_track_b_pass1.yaml'

    def _load_prompt(prompt_file: str) -> tuple[str, str, dict[str, Any]]:
        path = (
            Path(__file__).parent.parent.parent.parent
            / "foundation" / "prompts" / "_defaults"
            / prompt_file
        )
        if not path.exists():
>           raise FileNotFoundError(f"Prompt not found: {path}")
E           FileNotFoundError: Prompt not found: /home/ll/Public/QuantNodes/QuantNodes/foundation/prompts/_defaults/repro_extract_track_b_pass1.yaml

QuantNodes/research/paper_understanding/llm_extraction/track_b.py:191: FileNotFoundError
_________________ TestRunPass1.test_consecutive_zero_new_stops _________________

self = <test_track_b_multiturn.TestRunPass1 object at 0x78893a66b0d0>

    def test_consecutive_zero_new_stops(self):
        """Three consecutive rounds with zero new → stop (F1: MAX_CONSECUTIVE_ZERO 2→3)."""
        plan = PlanResult(
            paper_id="test",
            schema_choice="factor",
            n_signals_estimate=30,
            confidence=0.95,
            token_budget={"track_b_pass1": PASS1_MAX_TOKENS_DEFAULT},
            success=True,
        )
        client = RoundSequenceFakeLLM([
            single_round_response(20, done=False),  # 20 new
            "{}",  # 0 new (consecutive_zero=1)
            "{}",  # 0 new (consecutive_zero=2)
            "{}",  # 0 new (consecutive_zero=3) → stop after 4 rounds
        ])
>       stubs, latency, n_calls = _run_pass1(
            client, plan, "test", "dummy text",
        )

tests/research/test_track_b_multiturn.py:200: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/paper_understanding/llm_extraction/track_b.py:876: in _run_pass1
    system_text, user_template, params = _load_prompt(PROMPT_PASS1)
                                         ^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

prompt_file = 'repro_extract_track_b_pass1.yaml'

    def _load_prompt(prompt_file: str) -> tuple[str, str, dict[str, Any]]:
        path = (
            Path(__file__).parent.parent.parent.parent
            / "foundation" / "prompts" / "_defaults"
            / prompt_file
        )
        if not path.exists():
>           raise FileNotFoundError(f"Prompt not found: {path}")
E           FileNotFoundError: Prompt not found: /home/ll/Public/QuantNodes/QuantNodes/foundation/prompts/_defaults/repro_extract_track_b_pass1.yaml

QuantNodes/research/paper_understanding/llm_extraction/track_b.py:191: FileNotFoundError
_______________________ TestRunPass1.test_max_rounds_cap _______________________

self = <test_track_b_multiturn.TestRunPass1 object at 0x78893a66b690>

    def test_max_rounds_cap(self):
        """Stop when hits MAX_ROUNDS even if not done.
    
        If every round has one NEW signal (no duplicates), stops at exactly MAX_ROUNDS.
        """
        plan = PlanResult(
            paper_id="test",
            schema_choice="factor",
            n_signals_estimate=200,
            confidence=0.95,
            token_budget={"track_b_pass1": PASS1_MAX_TOKENS_DEFAULT},
            success=True,
        )
        # every round different name, so new every time → stop at MAX_ROUNDS
        responses = []
        for i in range(MAX_ROUNDS + 2):
            responses.append(json.dumps({
                "signals": [{"name": f"Alpha#{i+1}", "formula": "f"}],
                "done": False,
            }))
        client = RoundSequenceFakeLLM(responses)
>       stubs, latency, n_calls = _run_pass1(
            client, plan, "test", "dummy text",
        )

tests/research/test_track_b_multiturn.py:227: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/paper_understanding/llm_extraction/track_b.py:876: in _run_pass1
    system_text, user_template, params = _load_prompt(PROMPT_PASS1)
                                         ^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

prompt_file = 'repro_extract_track_b_pass1.yaml'

    def _load_prompt(prompt_file: str) -> tuple[str, str, dict[str, Any]]:
        path = (
            Path(__file__).parent.parent.parent.parent
            / "foundation" / "prompts" / "_defaults"
            / prompt_file
        )
        if not path.exists():
>           raise FileNotFoundError(f"Prompt not found: {path}")
E           FileNotFoundError: Prompt not found: /home/ll/Public/QuantNodes/QuantNodes/foundation/prompts/_defaults/repro_extract_track_b_pass1.yaml

QuantNodes/research/paper_understanding/llm_extraction/track_b.py:191: FileNotFoundError
_____________________ TestRunPass1.test_dedup_keeps_unique _____________________

self = <test_track_b_multiturn.TestRunPass1 object at 0x78893a66bc50>

    def test_dedup_keeps_unique(self):
        """Duplicate names are deduped across rounds, only first kept."""
        plan = PlanResult(
            paper_id="test",
            schema_choice="factor",
            n_signals_estimate=19,
            confidence=0.95,
            token_budget={"track_b_pass1": PASS1_MAX_TOKENS_DEFAULT},
            success=True,
        )
        # round 1: 1-10; round 2: repeats 10 + 11-19 → 19 unique
        resp1 = json.dumps({
            "signals": [{"name": f"Alpha#{i}", "formula": str(i)} for i in range(1, 11)],
            "done": False,
        })
        resp2 = json.dumps({
            "signals": [
                {"name": "Alpha#10", "formula": "10"},  # duplicate
                *[{"name": f"Alpha#{i}", "formula": str(i)} for i in range(11, 20)],
            ],
            "done": True,
        })
        client = RoundSequenceFakeLLM([resp1, resp2])
>       stubs, latency, n_calls = _run_pass1(
            client, plan, "test", "dummy text",
        )

tests/research/test_track_b_multiturn.py:256: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/paper_understanding/llm_extraction/track_b.py:876: in _run_pass1
    system_text, user_template, params = _load_prompt(PROMPT_PASS1)
                                         ^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

prompt_file = 'repro_extract_track_b_pass1.yaml'

    def _load_prompt(prompt_file: str) -> tuple[str, str, dict[str, Any]]:
        path = (
            Path(__file__).parent.parent.parent.parent
            / "foundation" / "prompts" / "_defaults"
            / prompt_file
        )
        if not path.exists():
>           raise FileNotFoundError(f"Prompt not found: {path}")
E           FileNotFoundError: Prompt not found: /home/ll/Public/QuantNodes/QuantNodes/foundation/prompts/_defaults/repro_extract_track_b_pass1.yaml

QuantNodes/research/paper_understanding/llm_extraction/track_b.py:191: FileNotFoundError
_______________ TestRunPass1.test_empty_response_does_not_crash ________________

self = <test_track_b_multiturn.TestRunPass1 object at 0x78893a66b210>

    def test_empty_response_does_not_crash(self):
        plan = PlanResult(
            paper_id="test",
            schema_choice="factor",
            n_signals_estimate=101,
            confidence=0.95,
            token_budget={"track_b_pass1": PASS1_MAX_TOKENS_DEFAULT},
            success=True,
        )
        client = RoundSequenceFakeLLM(['{"signals":[],"done":false}'])
>       stubs, latency, n_calls = _run_pass1(
            client, plan, "test", "dummy text",
        )

tests/research/test_track_b_multiturn.py:275: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/paper_understanding/llm_extraction/track_b.py:876: in _run_pass1
    system_text, user_template, params = _load_prompt(PROMPT_PASS1)
                                         ^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

prompt_file = 'repro_extract_track_b_pass1.yaml'

    def _load_prompt(prompt_file: str) -> tuple[str, str, dict[str, Any]]:
        path = (
            Path(__file__).parent.parent.parent.parent
            / "foundation" / "prompts" / "_defaults"
            / prompt_file
        )
        if not path.exists():
>           raise FileNotFoundError(f"Prompt not found: {path}")
E           FileNotFoundError: Prompt not found: /home/ll/Public/QuantNodes/QuantNodes/foundation/prompts/_defaults/repro_extract_track_b_pass1.yaml

QuantNodes/research/paper_understanding/llm_extraction/track_b.py:191: FileNotFoundError
________________ TestInitFactorWiki.test_init_creates_structure ________________

self = <test_wiki.TestInitFactorWiki object at 0x78893a6afa90>

    def test_init_creates_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wiki_path = os.path.join(tmpdir, "new_wiki")
>           init_factor_wiki(wiki_path)

tests/research/test_wiki.py:93: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
QuantNodes/research/wiki.py:194: in init_factor_wiki
    wiki.init()
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:35: in init
    return self._create_new_init(agent, force, merge, overwrite)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:86: in _create_new_init
    self._handle_wiki_md_schema(created, skipped, warnings, force, merge)
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:171: in _handle_wiki_md_schema
    self.wiki_md_file.write_text(self._generate_wiki_md())
                                 ^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/wiki_mixin_init.py:293: in _generate_wiki_md
    return registry.render_document("wiki_schema", version=self._get_version())
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:92: in render_document
    template = self._load_template(prompt_name)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <llmwikify.core.prompt_registry.PromptRegistry object at 0x7888b8b7da50>
prompt_name = 'wiki_schema'

    def _load_template(self, prompt_name: str) -> PromptTemplate:
        """Load and cache a prompt template from file."""
        if prompt_name in self._cache:
            return self._cache[prompt_name]
    
        yaml_path = self._find_prompt_file(prompt_name)
        if not yaml_path:
>           raise FileNotFoundError(
                f"Prompt template '{prompt_name}' not found in "
                f"{self._defaults_dir} or {self.custom_dir}"
            )
E           FileNotFoundError: Prompt template 'wiki_schema' not found in /home/ll/Public/QuantNodes/.venv-mig/lib/python3.11/site-packages/llmwikify/prompts/_defaults or None

.venv-mig/lib/python3.11/site-packages/llmwikify/core/prompt_registry.py:113: FileNotFoundError
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
  /home/ll/Public/QuantNodes/QuantNodes/research/paper_understanding/llm_extraction/orchestrator.py:256: DeprecationWarning: reproduction.common.llm_factory.build_default_client is deprecated; use QuantNodes.research.common.llm.client.build_llm_client instead. This wrapper will be removed in a future release.
    client = llm_client or build_default_client()

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
=========================== short test summary info ============================
FAILED tests/research/test_llm_factory.py::TestBuildDefaultClient::test_builds_client_with_valid_config
FAILED tests/research/test_orchestrator.py::TestOrchestratorEnd2End::test_defer_section_detector_continues_with_no_sections
FAILED tests/research/test_orchestrator.py::TestOrchestratorEnd2End::test_no_defer_no_queue
FAILED tests/research/test_retry_integration.py::TestTrackATier1Retry::test_retries_on_transient_failure
FAILED tests/research/test_retry_integration.py::TestTrackATier2Retry::test_retries_on_transient_failure
FAILED tests/research/test_retry_integration.py::TestTrackATier2Retry::test_all_tier2_prompts_succeed
FAILED tests/research/test_retry_integration.py::TestTrackBPass1MultiTurnRetry::test_retries_on_transient_failure
FAILED tests/research/test_retry_integration.py::TestTrackBPass2Retry::test_retries_on_transient_failure
FAILED tests/research/test_track_b_adaptive_helpers.py::TestAdaptiveMultiTurn::test_completes_batch_with_sufficient_context
FAILED tests/research/test_track_b_adaptive_helpers.py::TestAdaptiveMultiTurn::test_handles_need_more_context_with_supplement
FAILED tests/research/test_track_b_adaptive_helpers.py::TestAdaptiveMultiTurn::test_max_supplements_exceeded_marks_failed
FAILED tests/research/test_track_b_adaptive_helpers.py::TestAdaptiveMultiTurn::test_json_parse_failure_continues_to_next_round
FAILED tests/research/test_track_b_adaptive_helpers.py::TestAdaptiveMultiTurn::test_legacy_single_factor_format_supported
FAILED tests/research/test_track_b_adaptive_helpers.py::TestAdaptiveMultiTurn::test_resume_skips_existing_details
FAILED tests/research/test_track_b_adaptive_helpers.py::TestAdaptiveMultiTurn::test_history_trimmed_after_max_messages
FAILED tests/research/test_track_b_hybrid.py::TestHybridPass2Orchestration::test_hybrid_phase1_phase2_merge
FAILED tests/research/test_track_b_hybrid.py::TestHybridPass2Orchestration::test_hybrid_no_shallow_skips_adaptive
FAILED tests/research/test_track_b_hybrid.py::TestSupplementPrompt::test_supplement_prompt_loads
FAILED tests/research/test_track_b_hybrid.py::TestSupplementPrompt::test_supplement_prompt_requires_depth
FAILED tests/research/test_track_b_hybrid.py::TestSupplementPrompt::test_supplement_prompt_differs_from_standard
FAILED tests/research/test_track_b_multiturn.py::TestRunPass1::test_one_round_all_done
FAILED tests/research/test_track_b_multiturn.py::TestRunPass1::test_two_rounds_continuation
FAILED tests/research/test_track_b_multiturn.py::TestRunPass1::test_done_by_llm_before_count
FAILED tests/research/test_track_b_multiturn.py::TestRunPass1::test_consecutive_zero_new_stops
FAILED tests/research/test_track_b_multiturn.py::TestRunPass1::test_max_rounds_cap
FAILED tests/research/test_track_b_multiturn.py::TestRunPass1::test_dedup_keeps_unique
FAILED tests/research/test_track_b_multiturn.py::TestRunPass1::test_empty_response_does_not_crash
FAILED tests/research/test_wiki.py::TestInitFactorWiki::test_init_creates_structure
ERROR tests/research/test_auto_research.py::TestAutoResearcher::test_run_basic
ERROR tests/research/test_auto_research.py::TestAutoResearcher::test_run_generates_report
ERROR tests/research/test_auto_research.py::TestAutoResearcher::test_mine_single_factor
ERROR tests/research/test_auto_research.py::TestAutoResearcher::test_store_to_wiki
ERROR tests/research/test_auto_research.py::TestAutoResearcherEdge::test_run_with_empty_data
ERROR tests/research/test_auto_research.py::TestAutoResearcherEdge::test_run_with_max_factors_zero
ERROR tests/research/test_auto_research.py::TestAutoResearcherEdge::test_run_with_custom_eval_config
ERROR tests/research/test_auto_research.py::TestAutoResearcherEdge::test_run_use_mcts_flag
ERROR tests/research/test_auto_research.py::TestAutoResearcherEdge::test_mine_single_factor_invalid_formula
ERROR tests/research/test_auto_research.py::TestAutoResearcherEdge::test_mine_single_factor_stores_to_wiki
ERROR tests/research/test_paper_api.py::test_start_returns_session_id
ERROR tests/research/test_paper_api.py::test_start_default_wiki_id
ERROR tests/research/test_paper_api.py::test_start_explicit_wiki_id
ERROR tests/research/test_paper_api.py::test_start_invalid_source_type
ERROR tests/research/test_paper_api.py::test_list_empty
ERROR tests/research/test_paper_api.py::test_list_with_sessions
ERROR tests/research/test_paper_api.py::test_list_filter_status
ERROR tests/research/test_paper_api.py::test_list_raw_empty
ERROR tests/research/test_paper_api.py::test_list_raw
ERROR tests/research/test_paper_api.py::test_upload_pdf
ERROR tests/research/test_paper_api.py::test_upload_rejects_non_pdf
ERROR tests/research/test_paper_api.py::test_upload_rejects_empty
ERROR tests/research/test_paper_api.py::test_status_returns_session_and_events
ERROR tests/research/test_paper_api.py::test_status_not_found
ERROR tests/research/test_paper_api.py::test_legacy_paper_id
ERROR tests/research/test_paper_api.py::test_legacy_artifacts
ERROR tests/research/test_paper_api.py::test_delete_session
ERROR tests/research/test_paper_api.py::test_delete_not_found
ERROR tests/research/test_report_reproducer.py::TestParsePdf::test_parse_pdf_fallback
ERROR tests/research/test_report_reproducer.py::TestRuleBasedExtract::test_extract_formula_patterns
ERROR tests/research/test_report_reproducer.py::TestRuleBasedExtract::test_extract_no_match
ERROR tests/research/test_report_reproducer.py::TestRuleBasedExtract::test_extract_dedup
ERROR tests/research/test_report_reproducer.py::TestLLMExtract::test_llm_extract_success
ERROR tests/research/test_report_reproducer.py::TestLLMExtract::test_llm_extract_fallback_on_error
ERROR tests/research/test_report_reproducer.py::TestVerifyFactor::test_verify_factor_valid
ERROR tests/research/test_report_reproducer.py::TestVerifyFactor::test_verify_no_formula
ERROR tests/research/test_report_reproducer.py::TestVerifyFactor::test_verify_invalid_formula
ERROR tests/research/test_report_reproducer.py::TestWikiStorage::test_store_verified_factor
ERROR tests/research/test_report_reproducer.py::TestWikiStorage::test_store_pending_logic
ERROR tests/research/test_report_reproducer.py::TestReportGeneration::test_generate_report
ERROR tests/research/test_report_reproducer.py::TestProcessE2E::test_process_with_rule_extract
ERROR tests/research/test_report_reproducer.py::TestProcessE2E::test_process_without_data
ERROR tests/research/test_report_reproducer.py::TestExtractLogicFromText::test_extract_logic_from_text_public
ERROR tests/research/test_report_reproducer.py::TestExtractLogicFromText::test_extract_logic_from_text_empty
ERROR tests/research/test_report_reproducer.py::TestExtractLogicFromText::test_extract_logic_from_text_with_llm
ERROR tests/research/test_report_reproducer.py::TestReproductionReportDataclass::test_reproduction_report_fields
ERROR tests/research/test_report_reproducer.py::TestProcessWithInvalidPdf::test_process_invalid_pdf_path
ERROR tests/research/test_report_reproducer.py::TestProcessWithDataNoFormulas::test_process_with_data_no_formulas
ERROR tests/research/test_reproduction_api.py::test_start_returns_session_id
ERROR tests/research/test_reproduction_api.py::test_start_default_wiki_id
ERROR tests/research/test_reproduction_api.py::test_list_empty
ERROR tests/research/test_reproduction_api.py::test_list_with_sessions
ERROR tests/research/test_reproduction_api.py::test_get_session
ERROR tests/research/test_reproduction_api.py::test_get_not_found
ERROR tests/research/test_reproduction_api.py::test_artifacts_empty
ERROR tests/research/test_reproduction_api.py::test_delete_session
ERROR tests/research/test_reproduction_api.py::test_delete_not_found
ERROR tests/research/test_wiki.py::TestWikiFactorProxyCRUD::test_store_factor
ERROR tests/research/test_wiki.py::TestWikiFactorProxyCRUD::test_get_factor
ERROR tests/research/test_wiki.py::TestWikiFactorProxyCRUD::test_get_factor_not_found
ERROR tests/research/test_wiki.py::TestWikiFactorProxyCRUD::test_search_factors
ERROR tests/research/test_wiki.py::TestWikiFactorProxyCRUD::test_list_factors
ERROR tests/research/test_wiki.py::TestWikiFactorProxyCRUD::test_list_factors_with_filter
ERROR tests/research/test_wiki.py::TestWikiFactorProxyCRUD::test_list_factors_with_tags_filter
ERROR tests/research/test_wiki.py::TestWikiFactorProxyCRUD::test_list_factors_with_category_filter
ERROR tests/research/test_wiki.py::TestWikiFactorProxyCRUD::test_update_factor
ERROR tests/research/test_wiki.py::TestWikiFactorProxyCRUD::test_update_factor_not_found
ERROR tests/research/test_wiki.py::TestWikiFactorProxyCRUD::test_delete_factor
ERROR tests/research/test_wiki.py::TestWikiFactorProxyCRUD::test_delete_factor_not_found
ERROR tests/research/test_wiki.py::TestWikiLogicCRUD::test_store_logic - File...
ERROR tests/research/test_wiki.py::TestWikiLogicCRUD::test_get_logic - FileNo...
ERROR tests/research/test_wiki.py::TestWikiLogicCRUD::test_get_logic_not_found
ERROR tests/research/test_wiki.py::TestWikiLogicCRUD::test_search_logics - Fi...
ERROR tests/research/test_wiki.py::TestRelations::test_add_relation - FileNot...
ERROR tests/research/test_wiki.py::TestRelations::test_add_relation_invalid_type
ERROR tests/research/test_wiki.py::TestRelations::test_get_neighbors - FileNo...
ERROR tests/research/test_wiki.py::TestICIRSerialization::test_icir_fields_roundtrip
ERROR tests/research/test_wiki.py::TestStrategyYaml::test_strategy_yaml_roundtrip
ERROR tests/research/test_wiki.py::TestStatus::test_ping - FileNotFoundError:...
ERROR tests/research/test_wiki.py::TestStatus::test_status - FileNotFoundErro...
ERROR tests/research/test_wiki.py::TestWikiStrategyCRUD::test_store_strategy
ERROR tests/research/test_wiki.py::TestWikiStrategyCRUD::test_get_strategy - ...
ERROR tests/research/test_wiki.py::TestWikiStrategyCRUD::test_get_strategy_not_found
ERROR tests/research/test_wiki.py::TestWikiStrategyCRUD::test_list_strategies
ERROR tests/research/test_wiki.py::TestWikiStrategyCRUD::test_list_strategies_with_category_filter
ERROR tests/research/test_wiki.py::TestWikiStrategyCRUD::test_list_strategies_with_tags_filter
ERROR tests/research/test_wiki.py::TestWikiStrategyCRUD::test_strategy_roundtrip_with_yaml_and_json
ERROR tests/research/test_wiki.py::TestWikiReproductionCRUD::test_store_reproduction
ERROR tests/research/test_wiki.py::TestWikiReproductionCRUD::test_get_reproduction
ERROR tests/research/test_wiki.py::TestWikiReproductionCRUD::test_get_reproduction_not_found
ERROR tests/research/test_wiki.py::TestWikiReproductionCRUD::test_store_reproduction_slash_in_title
===== 28 failed, 1730 passed, 16 skipped, 27 warnings, 91 errors in 31.66s =====
