-- ============================================================
-- create_daily_views_czce.sql
-- CZCE 产品日频聚合 View
-- ============================================================

CREATE VIEW IF NOT EXISTS daily_quote.v_CF_daily AS
SELECT 'CZCE' AS Exchange, 'CF' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.CF FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_FG_daily AS
SELECT 'CZCE' AS Exchange, 'FG' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.FG FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_MA_daily AS
SELECT 'CZCE' AS Exchange, 'MA' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.MA FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_OI_daily AS
SELECT 'CZCE' AS Exchange, 'OI' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.OI FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_RM_daily AS
SELECT 'CZCE' AS Exchange, 'RM' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.RM FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_SA_daily AS
SELECT 'CZCE' AS Exchange, 'SA' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.SA FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_SR_daily AS
SELECT 'CZCE' AS Exchange, 'SR' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.SR FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_TA_daily AS
SELECT 'CZCE' AS Exchange, 'TA' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.TA FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_ZC_daily AS
SELECT 'CZCE' AS Exchange, 'ZC' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.ZC FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;
