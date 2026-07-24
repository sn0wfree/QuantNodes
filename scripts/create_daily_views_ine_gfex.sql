-- ============================================================
-- create_daily_views_ine.sql
-- INE 产品日频聚合 View
-- ============================================================

CREATE VIEW IF NOT EXISTS daily_quote.v_SC_daily AS
SELECT 'INE' AS Exchange, 'SC' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.SC FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_NR_daily AS
SELECT 'INE' AS Exchange, 'NR' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.NR FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_LU_daily AS
SELECT 'INE' AS Exchange, 'LU' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.LU FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

-- ============================================================
-- create_daily_views_gfex.sql
-- GFEX 产品日频聚合 View
-- ============================================================

CREATE VIEW IF NOT EXISTS daily_quote.v_SI_daily AS
SELECT 'GFEX' AS Exchange, 'SI' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.SI FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;

CREATE VIEW IF NOT EXISTS daily_quote.v_LC_daily AS
SELECT 'GFEX' AS Exchange, 'LC' AS Product, InstrumentID AS Symbol, TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open, max(LastPrice) AS High, min(LastPrice) AS Low, argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume, max(Turnover) - min(Turnover) AS Turnover, argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement, argMin(PreClosePrice, TradingDateTimeMS) AS PreClose, argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest,
    argMax(src_file_name, TradingDateTimeMS) AS SrcFile
FROM tick_quote.LC FINAL WHERE SessionType IN (2, 3) AND Exchange NOT LIKE '%\_night%'
GROUP BY InstrumentID, TradingDay;
