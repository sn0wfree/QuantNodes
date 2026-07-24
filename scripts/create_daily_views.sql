-- ============================================================
-- create_daily_views.sql
-- 为 tick_quote 中每个产品创建日频聚合 View
-- 使用方法: clickhouse-client --user updater --password updater --multiquery < scripts/create_daily_views.sql
-- ============================================================

-- CFFEX 产品
CREATE VIEW IF NOT EXISTS daily_quote.v_IC_daily AS
SELECT 'CFFEX' AS Exchange, 'IC' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.IC FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_IF_daily AS
SELECT 'CFFEX' AS Exchange, 'IF' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.IF FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_IH_daily AS
SELECT 'CFFEX' AS Exchange, 'IH' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.IH FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_IM_daily AS
SELECT 'CFFEX' AS Exchange, 'IM' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.IM FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_T_daily AS
SELECT 'CFFEX' AS Exchange, 'T' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.T FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_TF_daily AS
SELECT 'CFFEX' AS Exchange, 'TF' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.TF FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_TL_daily AS
SELECT 'CFFEX' AS Exchange, 'TL' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.TL FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_TS_daily AS
SELECT 'CFFEX' AS Exchange, 'TS' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.TS FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_HO_daily AS
SELECT 'CFFEX' AS Exchange, 'HO' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.HO FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_IO_daily AS
SELECT 'CFFEX' AS Exchange, 'IO' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.IO FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_MO_daily AS
SELECT 'CFFEX' AS Exchange, 'MO' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.MO FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;
