# FIFO Orderbook Prediction Market Challenge

## Overview

This challenge uses one shared FIFO limit order book for a single binary `YES`
contract.

Strategies submit passive limit orders once per timestep. The market is driven
by:

- a latent Gaussian score process with compound Poisson jumps
- an informed arbitrageur that knows the true probability
- uninformed retail market orders
- a static hidden-liquidity competitor

## Instrument

Trade a single `YES` share that settles to:

- `1.0` if `Z_T > 0`
- `0.0` otherwise

where `Z_t` is the latent score process.

Prices are integer percentage ticks:

- `price_ticks in {1, 2, ..., 99}`
- economic price = `price_ticks / 100`

Quantities are rounded down to the nearest `0.01`.

## Latent Process

The score process is:

```text
Z_{t+1} = Z_t + sigma * epsilon_t + sum_{k=1}^{N_t} J_{t,k}

epsilon_t ~ N(0, 1)
N_t ~ Poisson(lambda_jump)
J_{t,k} ~ N(mu_jump, sigma_jump^2)
```

The local runner uses these defaults:

- `n_steps = 2_000`
- `sigma = 0.02`
- `lambda_jump = 0.001`
- `mu_jump = 0.0`
- `sigma_jump = 0.75`

At step `t`, with `H = T - t` steps remaining, the informed fair value is:

```text
p_t = Pr(Z_T > 0 | Z_t)
```

Under the compound-Poisson Gaussian-jump model:

```text
p_t =
  sum_{n=0}^{infinity}
    Poi(n; lambda_jump * H)
    * Phi((Z_t + n * mu_jump) / sqrt(H * sigma^2 + n * sigma_jump^2))
```

## Book Mechanics

The book uses standard price-time priority:

- highest bid matches first
- lowest ask matches first
- same-price orders are FIFO
- resting orders remain until filled or cancelled

The participant manages passive limit orders through:

- `PlaceOrder(side, price_ticks, quantity)`
- `CancelOrder(order_id)`
- `CancelAll()`

## Competitor

The hidden competitor maintains a static ladder of resting orders around the
sampled starting probability.

- bids rest at every lower tick outside the competitor spread
- asks rest at every higher tick outside the competitor spread
- each price level uses the same notional size
- untouched resting competitor liquidity never recenters
- partially filled competitor orders stay in place until fully filled
- when a competitor order is fully filled, a replacement order appears at the
  start of the next step on the opposite side offset by `spread_ticks`

## Strategy API

The simulator calls:

```python
actions = strategy.on_step(state)
```

`state` contains:

- `step`
- `steps_remaining`
- `yes_inventory`
- `no_inventory`
- `cash`
- `reserved_cash`
- `free_cash`
- `competitor_best_bid_ticks`
- `competitor_best_ask_ticks`
- `buy_filled_quantity`
- `sell_filled_quantity`
- `own_orders`

`buy_filled_quantity` is the participant's total filled quantity on resting bids
from the previous step. `sell_filled_quantity` is the participant's total filled
quantity on resting asks from the previous step. Those values are aggregated
across all fills in the step.

## Event Loop

Each step runs in this order:

1. Refresh competitor replenishments from the previous step.
2. Call the participant strategy with current private state and the previous
   step's aggregate fill totals.
3. Apply participant cancels and new passive limit orders.
4. Advance the latent process and compute the true probability `p_t`.
5. Let the arbitrageur sweep stale quotes against `p_t`.
6. Generate retail market orders and match them against the book.
7. Record participant fills, edge, and competitor replenishments.

## Arbitrageur

The arbitrageur knows `p_t` and sends marketable flow only.

- buy every resting ask with `ask_price < p_t`
- sell into every resting bid with `bid_price > p_t`

## Retail Flow

Retail flow is exogenous to the latent state aside from sell-side
cash-to-quantity conversion.

- arrivals per step: `Poisson(lambda_retail)`
- side: `Bernoulli(0.5)`
- notional: `LogNormal(mu_ln, sigma_size)`
- buy order: spend `notional` cash on `YES`
- sell order: sell `notional / max(p_t, p_floor)` shares

## Scoring

The score is participant edge, not terminal PnL.

For each passive participant fill:

- participant buys `q` at price `x`: `edge = q * (p_t - x)`
- participant sells `q` at price `x`: `edge = q * (x - p_t)`

The primary score is mean participant edge across simulations.

## Collateral And Failure Rules

- participant starts with `$1000`, `0` YES, and `0` NO
- resting bids reserve `price * quantity` cash
- uncovered resting asks reserve `(1 - price) * quantity` cash
- the participant can mint one YES and one NO for `$1`
- invalid actions raise an exception and fail the simulation

## Current Scope

- one binary `YES` contract
- one shared FIFO order book
- one participant strategy
- one static hidden competitor ladder
- one informed arbitrageur
- uninformed retail market orders
- a passive-only participant API
- edge-based scoring
