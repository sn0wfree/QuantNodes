# 2026 年行情数据库数据字典 (futures_market_2026.duckdb)

切片口径: trade_date >= 2026-01-01 (行情至 2026-09-04, CH tick_quote 聚合).
共 455 张数据表 + 2 张元数据表.

## 与 full 库的差异

1. **内容为 2026 年切片** (管线 warmup 自 2025-10 起, 拼接/平滑边界正确).
2. **期权日线改由 CH tick_quote 聚合生成** (src_file='ch_tick_quote'),
   非交易所爬虫源; 剔除了套利合约(SP a&b / MSC / MS-C) / EFP / TAS /
   期转现(F) 等非标准合约.
3. **spot_basis_daily 由 akshare 重拉** (2025-07-22 起, 54 品种).
4. 滚动分位列 (vrp_percentile / iv_percentile / hv) 在 2026 年前段
   因窗口不足存在 NaN (切片库固有局限, warmup 自 2025-10).
5. 表级/列级说明同 DATA_DICTIONARY.md (full 库); 期权表结构:
   exchange/product/symbol/underlying/option_type/strike_price/
   trade_date/OHLCV/pre_*/src_file.

## 内嵌元数据

```sql
SELECT * FROM table_catalog;   -- 表级目录
SELECT * FROM data_dictionary; -- 列级字典
```

## 表级目录

| 表名 | 行数 | 日期范围 | 说明 |
|---|---:|---|---|
| `main_contract_mapping` | 13,437 | 2026-01-05~2026-09-04 | 日期×品种 → 主力合约映射 |
| `trading_calendar` | 164 | 2026-01-05~2026-09-04 | 交易日历 |
| `contract_specs` | 92 | — | 合约乘数/保证金等要素表 |
| `futures_rule_snapshot` | 158 | — | 交易所规则快照 |
| `cffex_ho_options_daily` | 28,724 | 2026-01-05~2026-09-04 | CFFEX CFFEX_HO 全部期权合约日行情 |
| `cffex_ic_options_daily` | 0 | — | CFFEX CFFEX_IC 全部期权合约日行情 |
| `cffex_if_options_daily` | 0 | — | CFFEX CFFEX_IF 全部期权合约日行情 |
| `cffex_ih_options_daily` | 0 | — | CFFEX CFFEX_IH 全部期权合约日行情 |
| `cffex_im_options_daily` | 0 | — | CFFEX CFFEX_IM 全部期权合约日行情 |
| `cffex_io_options_daily` | 39,144 | 2026-01-05~2026-09-04 | CFFEX CFFEX_IO 全部期权合约日行情 |
| `cffex_mo_options_daily` | 47,528 | 2026-01-05~2026-09-04 | CFFEX CFFEX_MO 全部期权合约日行情 |
| `cffex_t_options_daily` | 0 | — | CFFEX CFFEX_T 全部期权合约日行情 |
| `cffex_tf_options_daily` | 0 | — | CFFEX CFFEX_TF 全部期权合约日行情 |
| `cffex_tl_options_daily` | 0 | — | CFFEX CFFEX_TL 全部期权合约日行情 |
| `cffex_ts_options_daily` | 0 | — | CFFEX CFFEX_TS 全部期权合约日行情 |
| `czce_ap_options_daily` | 29,857 | 2026-01-05~2026-09-04 | CZCE CZCE_AP 全部期权合约日行情 |
| `czce_cf_options_daily` | 35,172 | 2026-01-05~2026-09-04 | CZCE CZCE_CF 全部期权合约日行情 |
| `czce_cj_options_daily` | 33,816 | 2026-01-05~2026-09-04 | CZCE CZCE_CJ 全部期权合约日行情 |
| `czce_cy_options_daily` | 0 | — | CZCE CZCE_CY 全部期权合约日行情 |
| `czce_fg_options_daily` | 61,660 | 2026-01-05~2026-09-04 | CZCE CZCE_FG 全部期权合约日行情 |
| `czce_jr_options_daily` | 0 | — | CZCE CZCE_JR 全部期权合约日行情 |
| `czce_lr_options_daily` | 0 | — | CZCE CZCE_LR 全部期权合约日行情 |
| `czce_ma_options_daily` | 64,424 | 2026-01-05~2026-09-04 | CZCE CZCE_MA 全部期权合约日行情 |
| `czce_oi_options_daily` | 28,042 | 2026-01-05~2026-09-04 | CZCE CZCE_OI 全部期权合约日行情 |
| `czce_pf_options_daily` | 36,052 | 2026-01-05~2026-09-04 | CZCE CZCE_PF 全部期权合约日行情 |
| `czce_pk_options_daily` | 20,037 | 2026-01-05~2026-09-04 | CZCE CZCE_PK 全部期权合约日行情 |
| `czce_pl_options_daily` | 23,012 | 2026-01-05~2026-09-04 | CZCE CZCE_PL 全部期权合约日行情 |
| `czce_pm_options_daily` | 0 | — | CZCE CZCE_PM 全部期权合约日行情 |
| `czce_pr_options_daily` | 32,341 | 2026-01-05~2026-09-04 | CZCE CZCE_PR 全部期权合约日行情 |
| `czce_px_options_daily` | 45,373 | 2026-01-05~2026-09-04 | CZCE CZCE_PX 全部期权合约日行情 |
| `czce_ri_options_daily` | 0 | — | CZCE CZCE_RI 全部期权合约日行情 |
| `czce_rm_options_daily` | 48,823 | 2026-01-05~2026-09-04 | CZCE CZCE_RM 全部期权合约日行情 |
| `czce_rs_options_daily` | 0 | — | CZCE CZCE_RS 全部期权合约日行情 |
| `czce_sa_options_daily` | 61,867 | 2026-01-05~2026-09-04 | CZCE CZCE_SA 全部期权合约日行情 |
| `czce_sf_options_daily` | 30,577 | 2026-01-05~2026-09-04 | CZCE CZCE_SF 全部期权合约日行情 |
| `czce_sh_options_daily` | 43,376 | 2026-01-05~2026-09-04 | CZCE CZCE_SH 全部期权合约日行情 |
| `czce_sm_options_daily` | 34,636 | 2026-01-05~2026-09-04 | CZCE CZCE_SM 全部期权合约日行情 |
| `czce_sr_options_daily` | 22,869 | 2026-01-05~2026-09-04 | CZCE CZCE_SR 全部期权合约日行情 |
| `czce_ta_options_daily` | 55,621 | 2026-01-05~2026-09-04 | CZCE CZCE_TA 全部期权合约日行情 |
| `czce_ur_options_daily` | 47,536 | 2026-01-05~2026-09-04 | CZCE CZCE_UR 全部期权合约日行情 |
| `czce_wh_options_daily` | 0 | — | CZCE CZCE_WH 全部期权合约日行情 |
| `czce_zc_options_daily` | 154 | 2026-01-05~2026-09-04 | CZCE CZCE_ZC 全部期权合约日行情 |
| `dce_a_options_daily` | 35,310 | 2026-01-05~2026-09-04 | DCE DCE_A 全部期权合约日行情 |
| `dce_b_options_daily` | 54,018 | 2026-01-05~2026-09-04 | DCE DCE_B 全部期权合约日行情 |
| `dce_bb_options_daily` | 0 | — | DCE DCE_BB 全部期权合约日行情 |
| `dce_bz_options_daily` | 59,869 | 2026-01-05~2026-09-04 | DCE DCE_BZ 全部期权合约日行情 |
| `dce_c_options_daily` | 40,964 | 2026-01-05~2026-09-04 | DCE DCE_C 全部期权合约日行情 |
| `dce_cs_options_daily` | 18,665 | 2026-01-05~2026-09-04 | DCE DCE_CS 全部期权合约日行情 |
| `dce_eb_options_daily` | 81,737 | 2026-01-05~2026-09-04 | DCE DCE_EB 全部期权合约日行情 |
| `dce_eg_options_daily` | 85,397 | 2026-01-05~2026-09-04 | DCE DCE_EG 全部期权合约日行情 |
| `dce_fb_options_daily` | 0 | — | DCE DCE_FB 全部期权合约日行情 |
| `dce_i_options_daily` | 81,713 | 2026-01-05~2026-09-04 | DCE DCE_I 全部期权合约日行情 |
| `dce_j_options_daily` | 1,010 | 2026-09-02~2026-09-04 | DCE DCE_J 全部期权合约日行情 |
| `dce_jd_options_daily` | 52,731 | 2026-01-05~2026-09-04 | DCE DCE_JD 全部期权合约日行情 |
| `dce_jm_options_daily` | 68,870 | 2026-01-16~2026-09-04 | DCE DCE_JM 全部期权合约日行情 |
| `dce_l_options_daily` | 78,958 | 2026-01-05~2026-09-04 | DCE DCE_L 全部期权合约日行情 |
| `dce_lg_options_daily` | 12,817 | 2026-01-05~2026-09-04 | DCE DCE_LG 全部期权合约日行情 |
| `dce_lh_options_daily` | 29,642 | 2026-01-05~2026-09-04 | DCE DCE_LH 全部期权合约日行情 |
| `dce_m_options_daily` | 31,050 | 2026-01-05~2026-09-04 | DCE DCE_M 全部期权合约日行情 |
| `dce_p_options_daily` | 75,831 | 2026-01-05~2026-09-04 | DCE DCE_P 全部期权合约日行情 |
| `dce_pg_options_daily` | 97,909 | 2026-01-05~2026-09-04 | DCE DCE_PG 全部期权合约日行情 |
| `dce_pp_options_daily` | 77,198 | 2026-01-05~2026-09-04 | DCE DCE_PP 全部期权合约日行情 |
| `dce_rr_options_daily` | 0 | — | DCE DCE_RR 全部期权合约日行情 |
| `dce_v_options_daily` | 74,608 | 2026-01-05~2026-09-04 | DCE DCE_V 全部期权合约日行情 |
| `dce_y_options_daily` | 45,521 | 2026-01-05~2026-09-04 | DCE DCE_Y 全部期权合约日行情 |
| `gfex_lc_options_daily` | 221,954 | 2026-01-05~2026-09-04 | GFEX GFEX_LC 全部期权合约日行情 |
| `gfex_pd_options_daily` | 78,677 | 2026-01-05~2026-09-04 | GFEX GFEX_PD 全部期权合约日行情 |
| `gfex_ps_options_daily` | 112,653 | 2026-01-05~2026-09-04 | GFEX GFEX_PS 全部期权合约日行情 |
| `gfex_pt_options_daily` | 89,686 | 2026-01-05~2026-09-04 | GFEX GFEX_PT 全部期权合约日行情 |
| `gfex_si_options_daily` | 110,304 | 2026-01-05~2026-09-04 | GFEX GFEX_SI 全部期权合约日行情 |
| `ine_bc_options_daily` | 11,277 | 2026-04-22~2026-09-04 | INE INE_BC 全部期权合约日行情 |
| `ine_ec_options_daily` | 0 | — | INE INE_EC 全部期权合约日行情 |
| `ine_lu_options_daily` | 0 | — | INE INE_LU 全部期权合约日行情 |
| `ine_nr_options_daily` | 20,295 | 2026-04-22~2026-09-04 | INE INE_NR 全部期权合约日行情 |
| `ine_sc_options_daily` | 63,636 | 2026-01-05~2026-09-04 | INE INE_SC 全部期权合约日行情 |
| `ips_ip_options_daily` | 0 | — | IPS IPS_IP 全部期权合约日行情 |
| `shfe_ad_options_daily` | 23,951 | 2026-01-05~2026-09-04 | SHFE SHFE_AD 全部期权合约日行情 |
| `shfe_ag_options_daily` | 323,359 | 2026-01-05~2026-09-04 | SHFE SHFE_AG 全部期权合约日行情 |
| `shfe_al_options_daily` | 79,237 | 2026-01-05~2026-09-04 | SHFE SHFE_AL 全部期权合约日行情 |
| `shfe_ao_options_daily` | 39,495 | 2026-01-05~2026-09-04 | SHFE SHFE_AO 全部期权合约日行情 |
| `shfe_au_options_daily` | 129,380 | 2026-01-05~2026-09-04 | SHFE SHFE_AU 全部期权合约日行情 |
| `shfe_br_options_daily` | 43,101 | 2026-01-05~2026-09-04 | SHFE SHFE_BR 全部期权合约日行情 |
| `shfe_bu_options_daily` | 41,681 | 2026-01-05~2026-09-04 | SHFE SHFE_BU 全部期权合约日行情 |
| `shfe_cu_options_daily` | 71,974 | 2026-01-05~2026-09-04 | SHFE SHFE_CU 全部期权合约日行情 |
| `shfe_fu_options_daily` | 72,548 | 2026-01-05~2026-09-04 | SHFE SHFE_FU 全部期权合约日行情 |
| `shfe_hc_options_daily` | 0 | — | SHFE SHFE_HC 全部期权合约日行情 |
| `shfe_ni_options_daily` | 47,297 | 2026-01-05~2026-09-04 | SHFE SHFE_NI 全部期权合约日行情 |
| `shfe_op_options_daily` | 14,020 | 2026-01-05~2026-09-04 | SHFE SHFE_OP 全部期权合约日行情 |
| `shfe_pb_options_daily` | 24,383 | 2026-01-05~2026-09-04 | SHFE SHFE_PB 全部期权合约日行情 |
| `shfe_rb_options_daily` | 26,434 | 2026-01-05~2026-09-04 | SHFE SHFE_RB 全部期权合约日行情 |
| `shfe_ru_options_daily` | 29,514 | 2026-01-05~2026-09-04 | SHFE SHFE_RU 全部期权合约日行情 |
| `shfe_sn_options_daily` | 45,527 | 2026-01-05~2026-09-04 | SHFE SHFE_SN 全部期权合约日行情 |
| `shfe_sp_options_daily` | 29,089 | 2026-01-05~2026-09-04 | SHFE SHFE_SP 全部期权合约日行情 |
| `shfe_ss_options_daily` | 0 | — | SHFE SHFE_SS 全部期权合约日行情 |
| `shfe_wr_options_daily` | 0 | — | SHFE SHFE_WR 全部期权合约日行情 |
| `shfe_zn_options_daily` | 38,920 | 2026-01-05~2026-09-04 | SHFE SHFE_ZN 全部期权合约日行情 |
| `smoothed_a_daily` | 164 | 2026-01-05~2026-09-04 | A 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_ad_daily` | 164 | 2026-01-05~2026-09-04 | AD 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_ag_daily` | 164 | 2026-01-05~2026-09-04 | AG 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_al_daily` | 164 | 2026-01-05~2026-09-04 | AL 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_ao_daily` | 164 | 2026-01-05~2026-09-04 | AO 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_ap_daily` | 163 | 2026-01-05~2026-09-04 | AP 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_au_daily` | 164 | 2026-01-05~2026-09-04 | AU 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_b_daily` | 164 | 2026-01-05~2026-09-04 | B 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_bb_daily` | 164 | 2026-01-05~2026-09-04 | BB 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_bc_daily` | 164 | 2026-01-05~2026-09-04 | BC 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_br_daily` | 164 | 2026-01-05~2026-09-04 | BR 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_bu_daily` | 164 | 2026-01-05~2026-09-04 | BU 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_bz_daily` | 164 | 2026-01-05~2026-09-04 | BZ 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_c_daily` | 164 | 2026-01-05~2026-09-04 | C 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_cf_daily` | 163 | 2026-01-05~2026-09-04 | CF 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_cj_daily` | 163 | 2026-01-05~2026-09-04 | CJ 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_cs_daily` | 164 | 2026-01-05~2026-09-04 | CS 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_cu_daily` | 164 | 2026-01-05~2026-09-04 | CU 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_cy_daily` | 163 | 2026-01-05~2026-09-04 | CY 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_eb_daily` | 164 | 2026-01-05~2026-09-04 | EB 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_ec_daily` | 164 | 2026-01-05~2026-09-04 | EC 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_eg_daily` | 164 | 2026-01-05~2026-09-04 | EG 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_fb_daily` | 164 | 2026-01-05~2026-09-04 | FB 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_fg_daily` | 163 | 2026-01-05~2026-09-04 | FG 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_fu_daily` | 164 | 2026-01-05~2026-09-04 | FU 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_hc_daily` | 164 | 2026-01-05~2026-09-04 | HC 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_i_daily` | 164 | 2026-01-05~2026-09-04 | I 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_ic_daily` | 164 | 2026-01-05~2026-09-04 | IC 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_if_daily` | 164 | 2026-01-05~2026-09-04 | IF 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_ih_daily` | 164 | 2026-01-05~2026-09-04 | IH 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_im_daily` | 164 | 2026-01-05~2026-09-04 | IM 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_j_daily` | 164 | 2026-01-05~2026-09-04 | J 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_jd_daily` | 164 | 2026-01-05~2026-09-04 | JD 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_jm_daily` | 164 | 2026-01-05~2026-09-04 | JM 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_jr_daily` | 0 | — | JR 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_l_daily` | 164 | 2026-01-05~2026-09-04 | L 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_lc_daily` | 164 | 2026-01-05~2026-09-04 | LC 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_lg_daily` | 164 | 2026-01-05~2026-09-04 | LG 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_lh_daily` | 164 | 2026-01-05~2026-09-04 | LH 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_lr_daily` | 0 | — | LR 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_lu_daily` | 164 | 2026-01-05~2026-09-04 | LU 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_m_daily` | 164 | 2026-01-05~2026-09-04 | M 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_ma_daily` | 163 | 2026-01-05~2026-09-04 | MA 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_ni_daily` | 164 | 2026-01-05~2026-09-04 | NI 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_nr_daily` | 164 | 2026-01-05~2026-09-04 | NR 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_oi_daily` | 163 | 2026-01-05~2026-09-04 | OI 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_op_daily` | 164 | 2026-01-05~2026-09-04 | OP 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_p_daily` | 164 | 2026-01-05~2026-09-04 | P 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_pb_daily` | 164 | 2026-01-05~2026-09-04 | PB 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_pd_daily` | 164 | 2026-01-05~2026-09-04 | PD 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_pf_daily` | 163 | 2026-01-05~2026-09-04 | PF 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_pg_daily` | 164 | 2026-01-05~2026-09-04 | PG 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_pk_daily` | 163 | 2026-01-05~2026-09-04 | PK 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_pl_daily` | 163 | 2026-01-05~2026-09-04 | PL 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_pm_daily` | 0 | — | PM 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_pp_daily` | 164 | 2026-01-05~2026-09-04 | PP 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_pr_daily` | 163 | 2026-01-05~2026-09-04 | PR 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_ps_daily` | 164 | 2026-01-05~2026-09-04 | PS 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_pt_daily` | 164 | 2026-01-05~2026-09-04 | PT 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_px_daily` | 164 | 2026-01-05~2026-09-04 | PX 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_rb_daily` | 164 | 2026-01-05~2026-09-04 | RB 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_ri_daily` | 0 | — | RI 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_rm_daily` | 164 | 2026-01-05~2026-09-04 | RM 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_rr_daily` | 164 | 2026-01-05~2026-09-04 | RR 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_rs_daily` | 164 | 2026-01-05~2026-09-04 | RS 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_ru_daily` | 164 | 2026-01-05~2026-09-04 | RU 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_sa_daily` | 164 | 2026-01-05~2026-09-04 | SA 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_sc_daily` | 164 | 2026-01-05~2026-09-04 | SC 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_sf_daily` | 164 | 2026-01-05~2026-09-04 | SF 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_sh_daily` | 164 | 2026-01-05~2026-09-04 | SH 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_si_daily` | 164 | 2026-01-05~2026-09-04 | SI 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_sm_daily` | 164 | 2026-01-05~2026-09-04 | SM 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_sn_daily` | 164 | 2026-01-05~2026-09-04 | SN 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_sp_daily` | 164 | 2026-01-05~2026-09-04 | SP 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_sr_daily` | 164 | 2026-01-05~2026-09-04 | SR 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_ss_daily` | 164 | 2026-01-05~2026-09-04 | SS 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_t_daily` | 164 | 2026-01-05~2026-09-04 | T 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_ta_daily` | 164 | 2026-01-05~2026-09-04 | TA 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_tf_daily` | 164 | 2026-01-05~2026-09-04 | TF 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_tl_daily` | 164 | 2026-01-05~2026-09-04 | TL 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_ts_daily` | 164 | 2026-01-05~2026-09-04 | TS 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_ur_daily` | 164 | 2026-01-05~2026-09-04 | UR 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_v_daily` | 164 | 2026-01-05~2026-09-04 | V 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_wh_daily` | 0 | — | WH 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_wr_daily` | 164 | 2026-01-05~2026-09-04 | WR 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_y_daily` | 164 | 2026-01-05~2026-09-04 | Y 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_zc_daily` | 0 | — | ZC 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `smoothed_zn_daily` | 164 | 2026-01-05~2026-09-04 | ZN 真主连平滑连续 (换月拼接平滑处理, 本项目标准回测口径) |
| `cffex_ho_futures_daily` | 0 | — | CFFEX CFFEX_HO 全部合约日行情 (交易所原始) |
| `cffex_ic_futures_daily` | 656 | 2026-01-05~2026-09-04 | CFFEX CFFEX_IC 全部合约日行情 (交易所原始) |
| `cffex_if_futures_daily` | 656 | 2026-01-05~2026-09-04 | CFFEX CFFEX_IF 全部合约日行情 (交易所原始) |
| `cffex_ih_futures_daily` | 656 | 2026-01-05~2026-09-04 | CFFEX CFFEX_IH 全部合约日行情 (交易所原始) |
| `cffex_im_futures_daily` | 656 | 2026-01-05~2026-09-04 | CFFEX CFFEX_IM 全部合约日行情 (交易所原始) |
| `cffex_io_futures_daily` | 0 | — | CFFEX CFFEX_IO 全部合约日行情 (交易所原始) |
| `cffex_mo_futures_daily` | 0 | — | CFFEX CFFEX_MO 全部合约日行情 (交易所原始) |
| `cffex_t_futures_daily` | 492 | 2026-01-05~2026-09-04 | CFFEX CFFEX_T 全部合约日行情 (交易所原始) |
| `cffex_tf_futures_daily` | 492 | 2026-01-05~2026-09-04 | CFFEX CFFEX_TF 全部合约日行情 (交易所原始) |
| `cffex_tl_futures_daily` | 492 | 2026-01-05~2026-09-04 | CFFEX CFFEX_TL 全部合约日行情 (交易所原始) |
| `cffex_ts_futures_daily` | 492 | 2026-01-05~2026-09-04 | CFFEX CFFEX_TS 全部合约日行情 (交易所原始) |
| `czce_ap_futures_daily` | 1,141 | 2026-01-05~2026-09-04 | CZCE CZCE_AP 全部合约日行情 (交易所原始) |
| `czce_cf_futures_daily` | 978 | 2026-01-05~2026-09-04 | CZCE CZCE_CF 全部合约日行情 (交易所原始) |
| `czce_cj_futures_daily` | 978 | 2026-01-05~2026-09-04 | CZCE CZCE_CJ 全部合约日行情 (交易所原始) |
| `czce_cy_futures_daily` | 1,956 | 2026-01-05~2026-09-04 | CZCE CZCE_CY 全部合约日行情 (交易所原始) |
| `czce_fg_futures_daily` | 1,956 | 2026-01-05~2026-09-04 | CZCE CZCE_FG 全部合约日行情 (交易所原始) |
| `czce_jr_futures_daily` | 978 | 2026-01-05~2026-09-04 | CZCE CZCE_JR 全部合约日行情 (交易所原始) |
| `czce_lr_futures_daily` | 10 | 2026-01-05~2026-01-16 | CZCE CZCE_LR 全部合约日行情 (交易所原始) |
| `czce_ma_futures_daily` | 1,956 | 2026-01-05~2026-09-04 | CZCE CZCE_MA 全部合约日行情 (交易所原始) |
| `czce_oi_futures_daily` | 978 | 2026-01-05~2026-09-04 | CZCE CZCE_OI 全部合约日行情 (交易所原始) |
| `czce_pf_futures_daily` | 1,956 | 2026-01-05~2026-09-04 | CZCE CZCE_PF 全部合约日行情 (交易所原始) |
| `czce_pk_futures_daily` | 1,141 | 2026-01-05~2026-09-04 | CZCE CZCE_PK 全部合约日行情 (交易所原始) |
| `czce_pl_futures_daily` | 1,956 | 2026-01-05~2026-09-04 | CZCE CZCE_PL 全部合约日行情 (交易所原始) |
| `czce_pm_futures_daily` | 978 | 2026-01-05~2026-09-04 | CZCE CZCE_PM 全部合约日行情 (交易所原始) |
| `czce_pr_futures_daily` | 1,956 | 2026-01-05~2026-09-04 | CZCE CZCE_PR 全部合约日行情 (交易所原始) |
| `czce_px_futures_daily` | 1,965 | 2026-01-05~2026-09-04 | CZCE CZCE_PX 全部合约日行情 (交易所原始) |
| `czce_ri_futures_daily` | 984 | 2026-01-05~2026-09-04 | CZCE CZCE_RI 全部合约日行情 (交易所原始) |
| `czce_rm_futures_daily` | 1,148 | 2026-01-05~2026-09-04 | CZCE CZCE_RM 全部合约日行情 (交易所原始) |
| `czce_rs_futures_daily` | 603 | 2026-01-05~2026-09-04 | CZCE CZCE_RS 全部合约日行情 (交易所原始) |
| `czce_sa_futures_daily` | 1,968 | 2026-01-05~2026-09-04 | CZCE CZCE_SA 全部合约日行情 (交易所原始) |
| `czce_sf_futures_daily` | 1,968 | 2026-01-05~2026-09-04 | CZCE CZCE_SF 全部合约日行情 (交易所原始) |
| `czce_sh_futures_daily` | 1,968 | 2026-01-05~2026-09-04 | CZCE CZCE_SH 全部合约日行情 (交易所原始) |
| `czce_sm_futures_daily` | 1,968 | 2026-01-05~2026-09-04 | CZCE CZCE_SM 全部合约日行情 (交易所原始) |
| `czce_sr_futures_daily` | 984 | 2026-01-05~2026-09-04 | CZCE CZCE_SR 全部合约日行情 (交易所原始) |
| `czce_ta_futures_daily` | 1,968 | 2026-01-05~2026-09-04 | CZCE CZCE_TA 全部合约日行情 (交易所原始) |
| `czce_ur_futures_daily` | 1,968 | 2026-01-05~2026-09-04 | CZCE CZCE_UR 全部合约日行情 (交易所原始) |
| `czce_wh_futures_daily` | 984 | 2026-01-05~2026-09-04 | CZCE CZCE_WH 全部合约日行情 (交易所原始) |
| `czce_zc_futures_daily` | 1,968 | 2026-01-05~2026-09-04 | CZCE CZCE_ZC 全部合约日行情 (交易所原始) |
| `dce_a_futures_daily` | 981 | 2026-01-05~2026-09-04 | DCE DCE_A 全部合约日行情 (交易所原始) |
| `dce_b_futures_daily` | 1,935 | 2026-01-05~2026-09-04 | DCE DCE_B 全部合约日行情 (交易所原始) |
| `dce_bb_futures_daily` | 815 | 2026-01-05~2026-09-04 | DCE DCE_BB 全部合约日行情 (交易所原始) |
| `dce_bz_futures_daily` | 1,900 | 2026-01-05~2026-09-04 | DCE DCE_BZ 全部合约日行情 (交易所原始) |
| `dce_c_futures_daily` | 982 | 2026-01-05~2026-09-04 | DCE DCE_C 全部合约日行情 (交易所原始) |
| `dce_cs_futures_daily` | 975 | 2026-01-05~2026-09-04 | DCE DCE_CS 全部合约日行情 (交易所原始) |
| `dce_eb_futures_daily` | 1,963 | 2026-01-05~2026-09-04 | DCE DCE_EB 全部合约日行情 (交易所原始) |
| `dce_eg_futures_daily` | 1,903 | 2026-01-05~2026-09-04 | DCE DCE_EG 全部合约日行情 (交易所原始) |
| `dce_fb_futures_daily` | 1,912 | 2026-01-05~2026-09-04 | DCE DCE_FB 全部合约日行情 (交易所原始) |
| `dce_i_futures_daily` | 1,942 | 2026-01-05~2026-09-04 | DCE DCE_I 全部合约日行情 (交易所原始) |
| `dce_j_futures_daily` | 1,923 | 2026-01-05~2026-09-04 | DCE DCE_J 全部合约日行情 (交易所原始) |
| `dce_jd_futures_daily` | 1,965 | 2026-01-05~2026-09-04 | DCE DCE_JD 全部合约日行情 (交易所原始) |
| `dce_jm_futures_daily` | 1,955 | 2026-01-05~2026-09-04 | DCE DCE_JM 全部合约日行情 (交易所原始) |
| `dce_l_futures_daily` | 1,966 | 2026-01-05~2026-09-04 | DCE DCE_L 全部合约日行情 (交易所原始) |
| `dce_lg_futures_daily` | 959 | 2026-01-05~2026-09-04 | DCE DCE_LG 全部合约日行情 (交易所原始) |
| `dce_lh_futures_daily` | 983 | 2026-01-05~2026-09-04 | DCE DCE_LH 全部合约日行情 (交易所原始) |
| `dce_m_futures_daily` | 1,310 | 2026-01-05~2026-09-04 | DCE DCE_M 全部合约日行情 (交易所原始) |
| `dce_p_futures_daily` | 1,960 | 2026-01-05~2026-09-04 | DCE DCE_P 全部合约日行情 (交易所原始) |
| `dce_pg_futures_daily` | 1,930 | 2026-01-05~2026-09-04 | DCE DCE_PG 全部合约日行情 (交易所原始) |
| `dce_pp_futures_daily` | 1,965 | 2026-01-05~2026-09-04 | DCE DCE_PP 全部合约日行情 (交易所原始) |
| `dce_rr_futures_daily` | 1,926 | 2026-01-05~2026-09-04 | DCE DCE_RR 全部合约日行情 (交易所原始) |
| `dce_v_futures_daily` | 1,968 | 2026-01-05~2026-09-04 | DCE DCE_V 全部合约日行情 (交易所原始) |
| `dce_y_futures_daily` | 1,306 | 2026-01-05~2026-09-04 | DCE DCE_Y 全部合约日行情 (交易所原始) |
| `gfex_lc_futures_daily` | 1,968 | 2026-01-05~2026-09-04 | GFEX GFEX_LC 全部合约日行情 (交易所原始) |
| `gfex_pd_futures_daily` | 888 | 2026-01-05~2026-09-04 | GFEX GFEX_PD 全部合约日行情 (交易所原始) |
| `gfex_ps_futures_daily` | 1,953 | 2026-01-05~2026-09-04 | GFEX GFEX_PS 全部合约日行情 (交易所原始) |
| `gfex_pt_futures_daily` | 888 | 2026-01-05~2026-09-04 | GFEX GFEX_PT 全部合约日行情 (交易所原始) |
| `gfex_si_futures_daily` | 1,968 | 2026-01-05~2026-09-04 | GFEX GFEX_SI 全部合约日行情 (交易所原始) |
| `ine_bc_futures_daily` | 1,940 | 2026-01-05~2026-09-04 | INE INE_BC 全部合约日行情 (交易所原始) |
| `ine_ec_futures_daily` | 1,275 | 2026-01-05~2026-09-04 | INE INE_EC 全部合约日行情 (交易所原始) |
| `ine_lu_futures_daily` | 1,965 | 2026-01-05~2026-09-04 | INE INE_LU 全部合约日行情 (交易所原始) |
| `ine_nr_futures_daily` | 1,958 | 2026-01-05~2026-09-04 | INE INE_NR 全部合约日行情 (交易所原始) |
| `ine_sc_futures_daily` | 3,279 | 2026-01-05~2026-09-04 | INE INE_SC 全部合约日行情 (交易所原始) |
| `ips_ip_futures_daily` | 0 | — | IPS IPS_IP 全部合约日行情 (交易所原始) |
| `shfe_ad_futures_daily` | 1,964 | 2026-01-05~2026-09-04 | SHFE SHFE_AD 全部合约日行情 (交易所原始) |
| `shfe_ag_futures_daily` | 1,968 | 2026-01-05~2026-09-04 | SHFE SHFE_AG 全部合约日行情 (交易所原始) |
| `shfe_al_futures_daily` | 1,968 | 2026-01-05~2026-09-04 | SHFE SHFE_AL 全部合约日行情 (交易所原始) |
| `shfe_ao_futures_daily` | 1,966 | 2026-01-05~2026-09-04 | SHFE SHFE_AO 全部合约日行情 (交易所原始) |
| `shfe_au_futures_daily` | 1,312 | 2026-01-05~2026-09-04 | SHFE SHFE_AU 全部合约日行情 (交易所原始) |
| `shfe_br_futures_daily` | 1,968 | 2026-01-05~2026-09-04 | SHFE SHFE_BR 全部合约日行情 (交易所原始) |
| `shfe_bu_futures_daily` | 2,621 | 2026-01-05~2026-09-04 | SHFE SHFE_BU 全部合约日行情 (交易所原始) |
| `shfe_cu_futures_daily` | 1,968 | 2026-01-05~2026-09-04 | SHFE SHFE_CU 全部合约日行情 (交易所原始) |
| `shfe_fu_futures_daily` | 1,967 | 2026-01-05~2026-09-04 | SHFE SHFE_FU 全部合约日行情 (交易所原始) |
| `shfe_hc_futures_daily` | 1,967 | 2026-01-05~2026-09-04 | SHFE SHFE_HC 全部合约日行情 (交易所原始) |
| `shfe_ni_futures_daily` | 1,968 | 2026-01-05~2026-09-04 | SHFE SHFE_NI 全部合约日行情 (交易所原始) |
| `shfe_op_futures_daily` | 1,959 | 2026-01-05~2026-09-04 | SHFE SHFE_OP 全部合约日行情 (交易所原始) |
| `shfe_pb_futures_daily` | 1,968 | 2026-01-05~2026-09-04 | SHFE SHFE_PB 全部合约日行情 (交易所原始) |
| `shfe_rb_futures_daily` | 1,967 | 2026-01-05~2026-09-04 | SHFE SHFE_RB 全部合约日行情 (交易所原始) |
| `shfe_ru_futures_daily` | 1,636 | 2026-01-05~2026-09-04 | SHFE SHFE_RU 全部合约日行情 (交易所原始) |
| `shfe_sn_futures_daily` | 1,968 | 2026-01-05~2026-09-04 | SHFE SHFE_SN 全部合约日行情 (交易所原始) |
| `shfe_sp_futures_daily` | 1,963 | 2026-01-05~2026-09-04 | SHFE SHFE_SP 全部合约日行情 (交易所原始) |
| `shfe_ss_futures_daily` | 1,968 | 2026-01-05~2026-09-04 | SHFE SHFE_SS 全部合约日行情 (交易所原始) |
| `shfe_wr_futures_daily` | 1,878 | 2026-01-05~2026-09-04 | SHFE SHFE_WR 全部合约日行情 (交易所原始) |
| `shfe_zn_futures_daily` | 1,968 | 2026-01-05~2026-09-04 | SHFE SHFE_ZN 全部合约日行情 (交易所原始) |
| `continuous_main_a_daily` | 164 | 2026-01-05~2026-09-04 | A 主力连续原始拼接 (未平滑) |
| `continuous_main_ad_daily` | 164 | 2026-01-05~2026-09-04 | AD 主力连续原始拼接 (未平滑) |
| `continuous_main_ag_daily` | 164 | 2026-01-05~2026-09-04 | AG 主力连续原始拼接 (未平滑) |
| `continuous_main_al_daily` | 164 | 2026-01-05~2026-09-04 | AL 主力连续原始拼接 (未平滑) |
| `continuous_main_ao_daily` | 164 | 2026-01-05~2026-09-04 | AO 主力连续原始拼接 (未平滑) |
| `continuous_main_ap_daily` | 163 | 2026-01-05~2026-09-04 | AP 主力连续原始拼接 (未平滑) |
| `continuous_main_au_daily` | 164 | 2026-01-05~2026-09-04 | AU 主力连续原始拼接 (未平滑) |
| `continuous_main_b_daily` | 164 | 2026-01-05~2026-09-04 | B 主力连续原始拼接 (未平滑) |
| `continuous_main_bb_daily` | 164 | 2026-01-05~2026-09-04 | BB 主力连续原始拼接 (未平滑) |
| `continuous_main_bc_daily` | 164 | 2026-01-05~2026-09-04 | BC 主力连续原始拼接 (未平滑) |
| `continuous_main_br_daily` | 164 | 2026-01-05~2026-09-04 | BR 主力连续原始拼接 (未平滑) |
| `continuous_main_bu_daily` | 164 | 2026-01-05~2026-09-04 | BU 主力连续原始拼接 (未平滑) |
| `continuous_main_bz_daily` | 164 | 2026-01-05~2026-09-04 | BZ 主力连续原始拼接 (未平滑) |
| `continuous_main_c_daily` | 164 | 2026-01-05~2026-09-04 | C 主力连续原始拼接 (未平滑) |
| `continuous_main_cf_daily` | 163 | 2026-01-05~2026-09-04 | CF 主力连续原始拼接 (未平滑) |
| `continuous_main_cj_daily` | 163 | 2026-01-05~2026-09-04 | CJ 主力连续原始拼接 (未平滑) |
| `continuous_main_cs_daily` | 164 | 2026-01-05~2026-09-04 | CS 主力连续原始拼接 (未平滑) |
| `continuous_main_cu_daily` | 164 | 2026-01-05~2026-09-04 | CU 主力连续原始拼接 (未平滑) |
| `continuous_main_cy_daily` | 163 | 2026-01-05~2026-09-04 | CY 主力连续原始拼接 (未平滑) |
| `continuous_main_eb_daily` | 164 | 2026-01-05~2026-09-04 | EB 主力连续原始拼接 (未平滑) |
| `continuous_main_ec_daily` | 164 | 2026-01-05~2026-09-04 | EC 主力连续原始拼接 (未平滑) |
| `continuous_main_eg_daily` | 164 | 2026-01-05~2026-09-04 | EG 主力连续原始拼接 (未平滑) |
| `continuous_main_fb_daily` | 164 | 2026-01-05~2026-09-04 | FB 主力连续原始拼接 (未平滑) |
| `continuous_main_fg_daily` | 163 | 2026-01-05~2026-09-04 | FG 主力连续原始拼接 (未平滑) |
| `continuous_main_fu_daily` | 164 | 2026-01-05~2026-09-04 | FU 主力连续原始拼接 (未平滑) |
| `continuous_main_hc_daily` | 164 | 2026-01-05~2026-09-04 | HC 主力连续原始拼接 (未平滑) |
| `continuous_main_i_daily` | 164 | 2026-01-05~2026-09-04 | I 主力连续原始拼接 (未平滑) |
| `continuous_main_ic_daily` | 164 | 2026-01-05~2026-09-04 | IC 主力连续原始拼接 (未平滑) |
| `continuous_main_if_daily` | 164 | 2026-01-05~2026-09-04 | IF 主力连续原始拼接 (未平滑) |
| `continuous_main_ih_daily` | 164 | 2026-01-05~2026-09-04 | IH 主力连续原始拼接 (未平滑) |
| `continuous_main_im_daily` | 164 | 2026-01-05~2026-09-04 | IM 主力连续原始拼接 (未平滑) |
| `continuous_main_j_daily` | 164 | 2026-01-05~2026-09-04 | J 主力连续原始拼接 (未平滑) |
| `continuous_main_jd_daily` | 164 | 2026-01-05~2026-09-04 | JD 主力连续原始拼接 (未平滑) |
| `continuous_main_jm_daily` | 164 | 2026-01-05~2026-09-04 | JM 主力连续原始拼接 (未平滑) |
| `continuous_main_jr_daily` | 0 | — | JR 主力连续原始拼接 (未平滑) |
| `continuous_main_l_daily` | 164 | 2026-01-05~2026-09-04 | L 主力连续原始拼接 (未平滑) |
| `continuous_main_lc_daily` | 164 | 2026-01-05~2026-09-04 | LC 主力连续原始拼接 (未平滑) |
| `continuous_main_lg_daily` | 164 | 2026-01-05~2026-09-04 | LG 主力连续原始拼接 (未平滑) |
| `continuous_main_lh_daily` | 164 | 2026-01-05~2026-09-04 | LH 主力连续原始拼接 (未平滑) |
| `continuous_main_lr_daily` | 0 | — | LR 主力连续原始拼接 (未平滑) |
| `continuous_main_lu_daily` | 164 | 2026-01-05~2026-09-04 | LU 主力连续原始拼接 (未平滑) |
| `continuous_main_m_daily` | 164 | 2026-01-05~2026-09-04 | M 主力连续原始拼接 (未平滑) |
| `continuous_main_ma_daily` | 163 | 2026-01-05~2026-09-04 | MA 主力连续原始拼接 (未平滑) |
| `continuous_main_ni_daily` | 164 | 2026-01-05~2026-09-04 | NI 主力连续原始拼接 (未平滑) |
| `continuous_main_nr_daily` | 164 | 2026-01-05~2026-09-04 | NR 主力连续原始拼接 (未平滑) |
| `continuous_main_oi_daily` | 163 | 2026-01-05~2026-09-04 | OI 主力连续原始拼接 (未平滑) |
| `continuous_main_op_daily` | 164 | 2026-01-05~2026-09-04 | OP 主力连续原始拼接 (未平滑) |
| `continuous_main_p_daily` | 164 | 2026-01-05~2026-09-04 | P 主力连续原始拼接 (未平滑) |
| `continuous_main_pb_daily` | 164 | 2026-01-05~2026-09-04 | PB 主力连续原始拼接 (未平滑) |
| `continuous_main_pd_daily` | 164 | 2026-01-05~2026-09-04 | PD 主力连续原始拼接 (未平滑) |
| `continuous_main_pf_daily` | 163 | 2026-01-05~2026-09-04 | PF 主力连续原始拼接 (未平滑) |
| `continuous_main_pg_daily` | 164 | 2026-01-05~2026-09-04 | PG 主力连续原始拼接 (未平滑) |
| `continuous_main_pk_daily` | 163 | 2026-01-05~2026-09-04 | PK 主力连续原始拼接 (未平滑) |
| `continuous_main_pl_daily` | 163 | 2026-01-05~2026-09-04 | PL 主力连续原始拼接 (未平滑) |
| `continuous_main_pm_daily` | 0 | — | PM 主力连续原始拼接 (未平滑) |
| `continuous_main_pp_daily` | 164 | 2026-01-05~2026-09-04 | PP 主力连续原始拼接 (未平滑) |
| `continuous_main_pr_daily` | 163 | 2026-01-05~2026-09-04 | PR 主力连续原始拼接 (未平滑) |
| `continuous_main_ps_daily` | 164 | 2026-01-05~2026-09-04 | PS 主力连续原始拼接 (未平滑) |
| `continuous_main_pt_daily` | 164 | 2026-01-05~2026-09-04 | PT 主力连续原始拼接 (未平滑) |
| `continuous_main_px_daily` | 164 | 2026-01-05~2026-09-04 | PX 主力连续原始拼接 (未平滑) |
| `continuous_main_rb_daily` | 164 | 2026-01-05~2026-09-04 | RB 主力连续原始拼接 (未平滑) |
| `continuous_main_ri_daily` | 0 | — | RI 主力连续原始拼接 (未平滑) |
| `continuous_main_rm_daily` | 164 | 2026-01-05~2026-09-04 | RM 主力连续原始拼接 (未平滑) |
| `continuous_main_rr_daily` | 164 | 2026-01-05~2026-09-04 | RR 主力连续原始拼接 (未平滑) |
| `continuous_main_rs_daily` | 164 | 2026-01-05~2026-09-04 | RS 主力连续原始拼接 (未平滑) |
| `continuous_main_ru_daily` | 164 | 2026-01-05~2026-09-04 | RU 主力连续原始拼接 (未平滑) |
| `continuous_main_sa_daily` | 164 | 2026-01-05~2026-09-04 | SA 主力连续原始拼接 (未平滑) |
| `continuous_main_sc_daily` | 164 | 2026-01-05~2026-09-04 | SC 主力连续原始拼接 (未平滑) |
| `continuous_main_sf_daily` | 164 | 2026-01-05~2026-09-04 | SF 主力连续原始拼接 (未平滑) |
| `continuous_main_sh_daily` | 164 | 2026-01-05~2026-09-04 | SH 主力连续原始拼接 (未平滑) |
| `continuous_main_si_daily` | 164 | 2026-01-05~2026-09-04 | SI 主力连续原始拼接 (未平滑) |
| `continuous_main_sm_daily` | 164 | 2026-01-05~2026-09-04 | SM 主力连续原始拼接 (未平滑) |
| `continuous_main_sn_daily` | 164 | 2026-01-05~2026-09-04 | SN 主力连续原始拼接 (未平滑) |
| `continuous_main_sp_daily` | 164 | 2026-01-05~2026-09-04 | SP 主力连续原始拼接 (未平滑) |
| `continuous_main_sr_daily` | 164 | 2026-01-05~2026-09-04 | SR 主力连续原始拼接 (未平滑) |
| `continuous_main_ss_daily` | 164 | 2026-01-05~2026-09-04 | SS 主力连续原始拼接 (未平滑) |
| `continuous_main_t_daily` | 164 | 2026-01-05~2026-09-04 | T 主力连续原始拼接 (未平滑) |
| `continuous_main_ta_daily` | 164 | 2026-01-05~2026-09-04 | TA 主力连续原始拼接 (未平滑) |
| `continuous_main_tf_daily` | 164 | 2026-01-05~2026-09-04 | TF 主力连续原始拼接 (未平滑) |
| `continuous_main_tl_daily` | 164 | 2026-01-05~2026-09-04 | TL 主力连续原始拼接 (未平滑) |
| `continuous_main_ts_daily` | 164 | 2026-01-05~2026-09-04 | TS 主力连续原始拼接 (未平滑) |
| `continuous_main_ur_daily` | 164 | 2026-01-05~2026-09-04 | UR 主力连续原始拼接 (未平滑) |
| `continuous_main_v_daily` | 164 | 2026-01-05~2026-09-04 | V 主力连续原始拼接 (未平滑) |
| `continuous_main_wh_daily` | 0 | — | WH 主力连续原始拼接 (未平滑) |
| `continuous_main_wr_daily` | 164 | 2026-01-05~2026-09-04 | WR 主力连续原始拼接 (未平滑) |
| `continuous_main_y_daily` | 164 | 2026-01-05~2026-09-04 | Y 主力连续原始拼接 (未平滑) |
| `continuous_main_zc_daily` | 0 | — | ZC 主力连续原始拼接 (未平滑) |
| `continuous_main_zn_daily` | 164 | 2026-01-05~2026-09-04 | ZN 主力连续原始拼接 (未平滑) |
| `gradual_a_daily` | 164 | 2026-01-05~2026-09-04 | A 渐进切换连续 (平滑的中间产物) |
| `gradual_ad_daily` | 164 | 2026-01-05~2026-09-04 | AD 渐进切换连续 (平滑的中间产物) |
| `gradual_ag_daily` | 164 | 2026-01-05~2026-09-04 | AG 渐进切换连续 (平滑的中间产物) |
| `gradual_al_daily` | 164 | 2026-01-05~2026-09-04 | AL 渐进切换连续 (平滑的中间产物) |
| `gradual_ao_daily` | 164 | 2026-01-05~2026-09-04 | AO 渐进切换连续 (平滑的中间产物) |
| `gradual_ap_daily` | 163 | 2026-01-05~2026-09-04 | AP 渐进切换连续 (平滑的中间产物) |
| `gradual_au_daily` | 164 | 2026-01-05~2026-09-04 | AU 渐进切换连续 (平滑的中间产物) |
| `gradual_b_daily` | 164 | 2026-01-05~2026-09-04 | B 渐进切换连续 (平滑的中间产物) |
| `gradual_bb_daily` | 164 | 2026-01-05~2026-09-04 | BB 渐进切换连续 (平滑的中间产物) |
| `gradual_bc_daily` | 164 | 2026-01-05~2026-09-04 | BC 渐进切换连续 (平滑的中间产物) |
| `gradual_br_daily` | 164 | 2026-01-05~2026-09-04 | BR 渐进切换连续 (平滑的中间产物) |
| `gradual_bu_daily` | 164 | 2026-01-05~2026-09-04 | BU 渐进切换连续 (平滑的中间产物) |
| `gradual_bz_daily` | 164 | 2026-01-05~2026-09-04 | BZ 渐进切换连续 (平滑的中间产物) |
| `gradual_c_daily` | 164 | 2026-01-05~2026-09-04 | C 渐进切换连续 (平滑的中间产物) |
| `gradual_cf_daily` | 163 | 2026-01-05~2026-09-04 | CF 渐进切换连续 (平滑的中间产物) |
| `gradual_cj_daily` | 163 | 2026-01-05~2026-09-04 | CJ 渐进切换连续 (平滑的中间产物) |
| `gradual_cs_daily` | 164 | 2026-01-05~2026-09-04 | CS 渐进切换连续 (平滑的中间产物) |
| `gradual_cu_daily` | 164 | 2026-01-05~2026-09-04 | CU 渐进切换连续 (平滑的中间产物) |
| `gradual_cy_daily` | 163 | 2026-01-05~2026-09-04 | CY 渐进切换连续 (平滑的中间产物) |
| `gradual_eb_daily` | 164 | 2026-01-05~2026-09-04 | EB 渐进切换连续 (平滑的中间产物) |
| `gradual_ec_daily` | 164 | 2026-01-05~2026-09-04 | EC 渐进切换连续 (平滑的中间产物) |
| `gradual_eg_daily` | 164 | 2026-01-05~2026-09-04 | EG 渐进切换连续 (平滑的中间产物) |
| `gradual_fb_daily` | 164 | 2026-01-05~2026-09-04 | FB 渐进切换连续 (平滑的中间产物) |
| `gradual_fg_daily` | 163 | 2026-01-05~2026-09-04 | FG 渐进切换连续 (平滑的中间产物) |
| `gradual_fu_daily` | 164 | 2026-01-05~2026-09-04 | FU 渐进切换连续 (平滑的中间产物) |
| `gradual_hc_daily` | 164 | 2026-01-05~2026-09-04 | HC 渐进切换连续 (平滑的中间产物) |
| `gradual_i_daily` | 164 | 2026-01-05~2026-09-04 | I 渐进切换连续 (平滑的中间产物) |
| `gradual_ic_daily` | 164 | 2026-01-05~2026-09-04 | IC 渐进切换连续 (平滑的中间产物) |
| `gradual_if_daily` | 164 | 2026-01-05~2026-09-04 | IF 渐进切换连续 (平滑的中间产物) |
| `gradual_ih_daily` | 164 | 2026-01-05~2026-09-04 | IH 渐进切换连续 (平滑的中间产物) |
| `gradual_im_daily` | 164 | 2026-01-05~2026-09-04 | IM 渐进切换连续 (平滑的中间产物) |
| `gradual_j_daily` | 164 | 2026-01-05~2026-09-04 | J 渐进切换连续 (平滑的中间产物) |
| `gradual_jd_daily` | 164 | 2026-01-05~2026-09-04 | JD 渐进切换连续 (平滑的中间产物) |
| `gradual_jm_daily` | 164 | 2026-01-05~2026-09-04 | JM 渐进切换连续 (平滑的中间产物) |
| `gradual_l_daily` | 164 | 2026-01-05~2026-09-04 | L 渐进切换连续 (平滑的中间产物) |
| `gradual_lc_daily` | 164 | 2026-01-05~2026-09-04 | LC 渐进切换连续 (平滑的中间产物) |
| `gradual_lg_daily` | 164 | 2026-01-05~2026-09-04 | LG 渐进切换连续 (平滑的中间产物) |
| `gradual_lh_daily` | 164 | 2026-01-05~2026-09-04 | LH 渐进切换连续 (平滑的中间产物) |
| `gradual_lu_daily` | 164 | 2026-01-05~2026-09-04 | LU 渐进切换连续 (平滑的中间产物) |
| `gradual_m_daily` | 164 | 2026-01-05~2026-09-04 | M 渐进切换连续 (平滑的中间产物) |
| `gradual_ma_daily` | 163 | 2026-01-05~2026-09-04 | MA 渐进切换连续 (平滑的中间产物) |
| `gradual_ni_daily` | 164 | 2026-01-05~2026-09-04 | NI 渐进切换连续 (平滑的中间产物) |
| `gradual_nr_daily` | 164 | 2026-01-05~2026-09-04 | NR 渐进切换连续 (平滑的中间产物) |
| `gradual_oi_daily` | 163 | 2026-01-05~2026-09-04 | OI 渐进切换连续 (平滑的中间产物) |
| `gradual_op_daily` | 164 | 2026-01-05~2026-09-04 | OP 渐进切换连续 (平滑的中间产物) |
| `gradual_p_daily` | 164 | 2026-01-05~2026-09-04 | P 渐进切换连续 (平滑的中间产物) |
| `gradual_pb_daily` | 164 | 2026-01-05~2026-09-04 | PB 渐进切换连续 (平滑的中间产物) |
| `gradual_pd_daily` | 164 | 2026-01-05~2026-09-04 | PD 渐进切换连续 (平滑的中间产物) |
| `gradual_pf_daily` | 163 | 2026-01-05~2026-09-04 | PF 渐进切换连续 (平滑的中间产物) |
| `gradual_pg_daily` | 164 | 2026-01-05~2026-09-04 | PG 渐进切换连续 (平滑的中间产物) |
| `gradual_pk_daily` | 163 | 2026-01-05~2026-09-04 | PK 渐进切换连续 (平滑的中间产物) |
| `gradual_pl_daily` | 163 | 2026-01-05~2026-09-04 | PL 渐进切换连续 (平滑的中间产物) |
| `gradual_pp_daily` | 164 | 2026-01-05~2026-09-04 | PP 渐进切换连续 (平滑的中间产物) |
| `gradual_pr_daily` | 163 | 2026-01-05~2026-09-04 | PR 渐进切换连续 (平滑的中间产物) |
| `gradual_ps_daily` | 164 | 2026-01-05~2026-09-04 | PS 渐进切换连续 (平滑的中间产物) |
| `gradual_pt_daily` | 164 | 2026-01-05~2026-09-04 | PT 渐进切换连续 (平滑的中间产物) |
| `gradual_px_daily` | 164 | 2026-01-05~2026-09-04 | PX 渐进切换连续 (平滑的中间产物) |
| `gradual_rb_daily` | 164 | 2026-01-05~2026-09-04 | RB 渐进切换连续 (平滑的中间产物) |
| `gradual_rm_daily` | 164 | 2026-01-05~2026-09-04 | RM 渐进切换连续 (平滑的中间产物) |
| `gradual_rr_daily` | 164 | 2026-01-05~2026-09-04 | RR 渐进切换连续 (平滑的中间产物) |
| `gradual_rs_daily` | 116 | 2026-01-05~2026-09-04 | RS 渐进切换连续 (平滑的中间产物) |
| `gradual_ru_daily` | 164 | 2026-01-05~2026-09-04 | RU 渐进切换连续 (平滑的中间产物) |
| `gradual_sa_daily` | 164 | 2026-01-05~2026-09-04 | SA 渐进切换连续 (平滑的中间产物) |
| `gradual_sc_daily` | 164 | 2026-01-05~2026-09-04 | SC 渐进切换连续 (平滑的中间产物) |
| `gradual_sf_daily` | 164 | 2026-01-05~2026-09-04 | SF 渐进切换连续 (平滑的中间产物) |
| `gradual_sh_daily` | 164 | 2026-01-05~2026-09-04 | SH 渐进切换连续 (平滑的中间产物) |
| `gradual_si_daily` | 164 | 2026-01-05~2026-09-04 | SI 渐进切换连续 (平滑的中间产物) |
| `gradual_sm_daily` | 164 | 2026-01-05~2026-09-04 | SM 渐进切换连续 (平滑的中间产物) |
| `gradual_sn_daily` | 164 | 2026-01-05~2026-09-04 | SN 渐进切换连续 (平滑的中间产物) |
| `gradual_sp_daily` | 164 | 2026-01-05~2026-09-04 | SP 渐进切换连续 (平滑的中间产物) |
| `gradual_sr_daily` | 164 | 2026-01-05~2026-09-04 | SR 渐进切换连续 (平滑的中间产物) |
| `gradual_ss_daily` | 164 | 2026-01-05~2026-09-04 | SS 渐进切换连续 (平滑的中间产物) |
| `gradual_t_daily` | 164 | 2026-01-05~2026-09-04 | T 渐进切换连续 (平滑的中间产物) |
| `gradual_ta_daily` | 164 | 2026-01-05~2026-09-04 | TA 渐进切换连续 (平滑的中间产物) |
| `gradual_tf_daily` | 164 | 2026-01-05~2026-09-04 | TF 渐进切换连续 (平滑的中间产物) |
| `gradual_tl_daily` | 164 | 2026-01-05~2026-09-04 | TL 渐进切换连续 (平滑的中间产物) |
| `gradual_ts_daily` | 164 | 2026-01-05~2026-09-04 | TS 渐进切换连续 (平滑的中间产物) |
| `gradual_ur_daily` | 164 | 2026-01-05~2026-09-04 | UR 渐进切换连续 (平滑的中间产物) |
| `gradual_v_daily` | 164 | 2026-01-05~2026-09-04 | V 渐进切换连续 (平滑的中间产物) |
| `gradual_wr_daily` | 164 | 2026-01-05~2026-09-04 | WR 渐进切换连续 (平滑的中间产物) |
| `gradual_y_daily` | 164 | 2026-01-05~2026-09-04 | Y 渐进切换连续 (平滑的中间产物) |
| `gradual_zn_daily` | 164 | 2026-01-05~2026-09-04 | ZN 渐进切换连续 (平滑的中间产物) |
| `spot_basis_daily` | 8,856 | 2026-01-05~2026-09-04 | 现货价-近月-主力 三口径基差 (symbol维度) |
| `volatility_hv_daily` | 13,437 | 2026-01-05~2026-09-04 | 88品种 HV(20/60/120日) 实现波动率 |
| `options_pcr_daily` | 10,744 | 2026-01-05~2026-09-04 | 65品种 Put/Call 比率 (成交量与持仓两个口径) |
| `term_structure_daily` | 137,392 | 2026-01-05~2026-09-04 | 88品种×多到期日合约结构. rank=1近月/2次近月...; days_to_expiry为年化carry分母; §25横截面carry与basis动量的数据源 |
| `vrp_daily` | 9,687 | 2026-01-05~2026-09-03 | 62品种 VRP = ATM隐含波动率 - 实现波动率(HV); 含 vrp/iv/hv 的滚动分位数与 regime 标签. PIT注意: IV为EOD口径 |
| `oi_crossover` | 13,389 | 2026-01-05~2026-09-04 | 88品种 主力/次主力 OI与价格 — 移仓换月进程监测 (main_symbol/next_symbol) |
| `cross_product_spread` | 4,259 | 2026-01-05~2026-09-04 | 跨品种价差序列 (研究用) |
| `inter_contract_spread` | 14,265 | 2026-01-05~2026-09-04 | 同品种跨期价差序列 (研究用) |
| `option_iv_daily` | 10,170 | 2026-01-05~2026-09-03 | 65品种 ATM IV + iv_percentile. 部分品种数据滞后 (RB/SC等2019-2022才上市期权) |
| `v_all_options` | 3,762,851 | 2026-01-05~2026-09-04 | 全品种期权宽表视图 |
| `v_all_futures` | 138,372 | 2026-01-05~2026-09-04 | 全品种期货宽表视图 |