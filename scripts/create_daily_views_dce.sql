-- ============================================================
-- create_daily_views_dce.sql
-- DCE 产品日频聚合 View
-- ============================================================

CREATE VIEW IF NOT EXISTS daily_quote.v_A_daily AS
SELECT 'DCE' AS Exchange, 'A' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.A FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_B_daily AS
SELECT 'DCE' AS Exchange, 'B' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.B FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_C_daily AS
SELECT 'DCE' AS Exchange, 'C' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.C FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_CS_daily AS
SELECT 'DCE' AS Exchange, 'CS' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.CS FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_EB_daily AS
SELECT 'DCE' AS Exchange, 'EB' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.EB FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_EG_daily AS
SELECT 'DCE' AS Exchange, 'EG' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.EG FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_I_daily AS
SELECT 'DCE' AS Exchange, 'I' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.I FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_J_daily AS
SELECT 'DCE' AS Exchange, 'J' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.J FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_JD_daily AS
SELECT 'DCE' AS Exchange, 'JD' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.JD FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_JM_daily AS
SELECT 'DCE' AS Exchange, 'JM' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.JM FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_L_daily AS
SELECT 'DCE' AS Exchange, 'L' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.L FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_M_daily AS
SELECT 'DCE' AS Exchange, 'M' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.M FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_P_daily AS
SELECT 'DCE' AS Exchange, 'P' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.P FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_PP_daily AS
SELECT 'DCE' AS Exchange, 'PP' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.PP FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_V_daily AS
SELECT 'DCE' AS Exchange, 'V' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.V FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_Y_daily AS
SELECT 'DCE' AS Exchange, 'Y' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.Y FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;
