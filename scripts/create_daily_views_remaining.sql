-- ============================================================
-- create_daily_views_remaining.sql
-- 剩余 40 个产品的日频聚合 View
-- ============================================================

-- SHFE 产品
CREATE VIEW IF NOT EXISTS daily_quote.v_AD_daily AS
SELECT 'SHFE' AS Exchange, 'AD' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.AD FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_AO_daily AS
SELECT 'SHFE' AS Exchange, 'AO' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.AO FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_BR_daily AS
SELECT 'SHFE' AS Exchange, 'BR' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.BR FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_OP_daily AS
SELECT 'SHFE' AS Exchange, 'OP' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.OP FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_SN_daily AS
SELECT 'SHFE' AS Exchange, 'SN' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.SN FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_SS_daily AS
SELECT 'SHFE' AS Exchange, 'SS' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.SS FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_WR_daily AS
SELECT 'SHFE' AS Exchange, 'WR' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.WR FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

-- DCE 产品
CREATE VIEW IF NOT EXISTS daily_quote.v_BB_daily AS
SELECT 'DCE' AS Exchange, 'BB' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.BB FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_BZ_daily AS
SELECT 'DCE' AS Exchange, 'BZ' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.BZ FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_FB_daily AS
SELECT 'DCE' AS Exchange, 'FB' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.FB FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_LG_daily AS
SELECT 'DCE' AS Exchange, 'LG' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.LG FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_LH_daily AS
SELECT 'DCE' AS Exchange, 'LH' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.LH FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_PG_daily AS
SELECT 'DCE' AS Exchange, 'PG' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.PG FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_RR_daily AS
SELECT 'DCE' AS Exchange, 'RR' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.RR FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

-- CZCE 产品
CREATE VIEW IF NOT EXISTS daily_quote.v_AP_daily AS
SELECT 'CZCE' AS Exchange, 'AP' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.AP FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_CJ_daily AS
SELECT 'CZCE' AS Exchange, 'CJ' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.CJ FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_CY_daily AS
SELECT 'CZCE' AS Exchange, 'CY' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.CY FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_JR_daily AS
SELECT 'CZCE' AS Exchange, 'JR' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.JR FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_LR_daily AS
SELECT 'CZCE' AS Exchange, 'LR' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.LR FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_PF_daily AS
SELECT 'CZCE' AS Exchange, 'PF' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.PF FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_PK_daily AS
SELECT 'CZCE' AS Exchange, 'PK' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.PK FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_PL_daily AS
SELECT 'CZCE' AS Exchange, 'PL' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.PL FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_PM_daily AS
SELECT 'CZCE' AS Exchange, 'PM' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.PM FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_PR_daily AS
SELECT 'CZCE' AS Exchange, 'PR' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.PR FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_PX_daily AS
SELECT 'CZCE' AS Exchange, 'PX' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.PX FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_RI_daily AS
SELECT 'CZCE' AS Exchange, 'RI' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.RI FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_RS_daily AS
SELECT 'CZCE' AS Exchange, 'RS' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.RS FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_SF_daily AS
SELECT 'CZCE' AS Exchange, 'SF' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.SF FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_SH_daily AS
SELECT 'CZCE' AS Exchange, 'SH' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.SH FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_SM_daily AS
SELECT 'CZCE' AS Exchange, 'SM' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.SM FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_UR_daily AS
SELECT 'CZCE' AS Exchange, 'UR' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.UR FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_WH_daily AS
SELECT 'CZCE' AS Exchange, 'WH' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.WH FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

-- INE 产品
CREATE VIEW IF NOT EXISTS daily_quote.v_BC_daily AS
SELECT 'INE' AS Exchange, 'BC' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.BC FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_EC_daily AS
SELECT 'INE' AS Exchange, 'EC' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.EC FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

-- GFEX 产品
CREATE VIEW IF NOT EXISTS daily_quote.v_PD_daily AS
SELECT 'GFEX' AS Exchange, 'PD' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.PD FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_PS_daily AS
SELECT 'GFEX' AS Exchange, 'PS' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.PS FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_PT_daily AS
SELECT 'GFEX' AS Exchange, 'PT' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.PT FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;
