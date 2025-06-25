CREATE DEFINER=`root`@`localhost` PROCEDURE `trades_PnL`()
BEGIN
drop table if exists kiteconnect.trades_pnl;
create table kiteconnect.trades_pnl as 
	with buy_trades as (
		Select tradingsymbol, 
				instrument_token, 
				order_id,
				sum(quantity) as quantity,
				min(date(order_timestamp)) as order_dt,
				sum(1.00*quantity*average_price) as total_investment
		from kiteconnect.trades
		where exchange <> 'NFO'
		and transaction_type = 'BUY'
		group by 1,2,3
	)

	, sell_trades as (
		Select tradingsymbol, 
				instrument_token, 
				order_id,
				sum(quantity) as quantity,
				min(date(order_timestamp)) as order_dt,
				sum(1.00*quantity*average_price) as total_investment
		from kiteconnect.trades
		where exchange <> 'NFO'
		and transaction_type = 'SELL'
		group by 1,2,3
	)
	, consolidated_trades as (
		Select a.tradingsymbol, 
				a.instrument_token,
				a.quantity as buy_quantity,
				a.order_dt as buy_order_dt,
				a.total_investment as buy_investment_amount,
				b.quantity as sell_quantity,
				b.order_dt as sell_order_dt,
				b.total_investment as sell_investment_amount,
				row_number() over (partition by a.order_id, b.order_id order by b.order_dt asc) as rnum
		from buy_trades as a
		left join sell_trades as b on a.instrument_token = b.instrument_token and b.order_dt > a.order_dt and b.order_dt <= date_add(a.order_dt, interval 20 day)
	)
	, latest_date as (
		Select max(date(timestamp)) as latest_dt
		from kiteconnect.historical_data_day
	)

	, latest_price as (
		Select instrument_token, close
		from kiteconnect.historical_data_day
		where timestamp = (Select latest_dt from latest_date)
	)
	Select a.tradingsymbol,
			a.instrument_token,
			a.buy_quantity,
			a.sell_quantity,
			a.buy_order_dt,
			a.sell_order_dt,
			a.buy_investment_amount,
			a.sell_investment_amount,
			case when sell_order_dt is null then 'Open' else 'Closed' end as trade_status,
			DATEDIFF(case when sell_order_dt is null then CURDATE() else sell_order_dt end,buy_order_dt) as holding_days,
			case when sell_order_dt is null then b.close*a.buy_quantity else sell_investment_amount end - buy_investment_amount as net_pnl,
			1.00*(case when sell_order_dt is null then b.close*a.buy_quantity else sell_investment_amount end - buy_investment_amount)/buy_quantity as net_pnl
	from consolidated_trades a
	left join latest_price b on a.instrument_token = b.instrument_token
	where rnum = 1;
END