"""
simulate.py - Creates the synthetic dataset (data/port_calls.csv).

WHY A SIMULATION AND NOT RANDOM NUMBERS?
If we wrote the label as a formula of the features (e.g. delay = 3*queue + 2*storm),
a model would simply rediscover our formula and the results would prove nothing.
Instead we simulate how a port works, hour by hour:

    ship arrives -> waits at anchorage if no suitable berth is free
                 -> gets a berth -> is unloaded -> leaves

Storms and strikes pause the port, planned maintenance takes a berth out of
service, and ship arrivals come in waves. The waiting time is an OUTCOME of
this process; we never type it in.

Then, for every ship, we record only what a planner could have known
48 hours BEFORE the ship arrived (queue, announced arrivals, weather forecast...)
plus the true outcome (how long the ship actually waited).

Run:  python -m src.simulate
"""
import numpy as np
import pandas as pd

from src import config as C


# ---------------------------------------------------------------------------
# Building blocks of the simulated world
# ---------------------------------------------------------------------------
def make_wind(rng, hours, start, exposure):
    """Hourly wind speed (km/h): normal breeze + random storms (more in monsoon)."""
    ts = pd.date_range(start, periods=hours, freq="h")
    months = ts.month.values
    # everyday wind: changes slowly from day to day, plus small hourly noise
    daily = 16 + 4 * exposure * np.isin(np.arange(1, 13), [6, 7, 8, 9])[months - 1]
    day_noise = np.repeat(rng.normal(0, 5, hours // 24 + 1), 24)[:hours]
    wind = daily + day_noise + rng.normal(0, 2, hours)

    # storms: random events, frequency depends on month and how exposed the port is
    h = 0
    while h < hours:
        rate_per_hour = 0.55 * exposure * C.MONTH_STORMS[months[h]] / (30 * 24)
        if rng.random() < rate_per_hour:
            duration = int(rng.uniform(18, 72))
            peak = rng.uniform(46, 85)
            shape = np.sin(np.linspace(0, np.pi, duration))      # ramps up then down
            end = min(h + duration, hours)
            wind[h:end] = np.maximum(wind[h:end], 20 + (peak - 20) * shape[: end - h])
            h += duration
        h += 1
    return np.clip(wind, 0, None)


def make_strikes(rng, hours, per_year):
    """Strike periods. Most are announced days ahead; some are sudden."""
    n = rng.poisson(per_year * hours / (365 * 24))
    strikes = []
    for _ in range(n):
        s = int(rng.uniform(0, hours))
        d = int(rng.uniform(12, 72))
        strikes.append(dict(start=s, end=min(s + d, hours),
                            announced=rng.random() > C.UNANNOUNCED_STRIKE_SHARE))
    return strikes


def make_maintenance(rng, hours, per_year):
    """Planned maintenance: one berth out of service for 3-10 days (known in advance)."""
    n = max(1, rng.poisson(per_year * hours / (365 * 24)))
    out = np.zeros(hours, dtype=int)
    for _ in range(n):
        s = int(rng.uniform(0, hours))
        d = int(rng.uniform(3, 10) * 24)
        out[s:s + d] = 1
    return out


def make_arrivals(rng, days, start, p):
    """Ship arrivals. Daily rate = base x season x weekday x 'wave' factor.

    The wave factor follows a slow random walk: when upstream ports or
    canals are disrupted, ships arrive bunched together for a while.
    """
    dates = pd.date_range(start, periods=days, freq="D")
    weekday = np.array([1.1, 1.1, 1.0, 1.0, 1.0, 0.95, 0.85])   # Mon..Sun
    log_wave, rows = 0.0, []
    for d, date in enumerate(dates):
        log_wave = 0.85 * log_wave + rng.normal(0, 0.18)
        rate = (p["arrivals_per_day"] * C.MONTH_TRAFFIC[date.month]
                * weekday[date.dayofweek] * np.exp(log_wave))
        for _ in range(rng.poisson(rate)):
            rows.append(d * 24 + int(rng.uniform(0, 24)))
    arrivals = np.sort(np.array(rows))

    # ship size (TEU) and how long unloading takes (hours of crane work)
    teu = np.clip(rng.lognormal(np.log(p["teu_median"]), 0.5, len(arrivals)),
                  600, p["teu_max"]).round()
    service_h = (12 + teu / 120) / p["productivity"] * rng.lognormal(0, 0.25, len(arrivals))
    return arrivals, teu, service_h


# ---------------------------------------------------------------------------
# The port itself: an hour-by-hour queue
# ---------------------------------------------------------------------------
def simulate_port(rng, name, p, hours):
    wind = make_wind(rng, hours, C.SIM_START, p["storm_exposure"])
    strikes = make_strikes(rng, hours, p["strikes_per_year"])
    strike_hour = np.zeros(hours, bool)
    for s in strikes:
        strike_hour[s["start"]:s["end"]] = True
    maint = make_maintenance(rng, hours, p["maintenance_per_year"])
    arrival_h, teu, service_h = make_arrivals(rng, hours // 24, C.SIM_START, p)

    n_ships = len(arrival_h)
    needs_deep = teu > C.DEEP_BERTH_TEU
    berth_start = np.full(n_ships, -1)
    departure = np.full(n_ships, -1)

    n_b = p["berths"]
    is_deep = np.array([i < p["deep_berths"] for i in range(n_b)])
    berth_ship = [-1] * n_b          # which ship occupies each berth (-1 = free)
    remaining = np.zeros(n_b)        # crane-hours of work left at each berth
    queue = []                       # ships waiting at anchorage, first-come-first-served
    queue_len = np.zeros(hours, int)
    busy_berths = np.zeros(hours, int)
    next_ship = 0

    for h in range(hours):
        port_open = wind[h] < C.STORM_CLOSURE_WIND_KMH and not strike_hour[h]

        # 1) ships at berth get unloaded (only while the port is working;
        #    half speed in rough weather)
        speed = 0.5 if wind[h] >= C.ROUGH_WEATHER_WIND_KMH else 1.0
        for b in range(n_b):
            if berth_ship[b] >= 0 and port_open:
                remaining[b] -= speed
                if remaining[b] <= 0:
                    departure[berth_ship[b]] = h
                    berth_ship[b] = -1

        # 2) new ships arrive and join the anchorage queue
        while next_ship < n_ships and arrival_h[next_ship] == h:
            queue.append(next_ship)
            next_ship += 1

        # 3) berthing: needs open port, a pilot (daylight only at some ports),
        #    and a free berth that is not under maintenance and is deep enough
        hour_of_day = h % 24
        can_berth = port_open and (not p["daylight_only"] or 6 <= hour_of_day < 18)
        if can_berth and queue:
            # maintenance always takes the last (shallow) berth out of service
            usable = [b for b in range(n_b) if not (maint[h] and b == n_b - 1)]
            for s in list(queue):
                free = [b for b in usable if berth_ship[b] < 0]
                if needs_deep[s]:
                    free = [b for b in free if is_deep[b]]
                else:   # small ships prefer shallow berths, keep deep ones for big ships
                    free = sorted(free, key=lambda b: is_deep[b])
                if free:
                    b = free[0]
                    berth_ship[b], remaining[b] = s, service_h[s]
                    berth_start[s] = h
                    queue.remove(s)

        queue_len[h] = len(queue)
        busy_berths[h] = sum(x >= 0 for x in berth_ship)

    ships = pd.DataFrame(dict(port=name, arrival_h=arrival_h, teu=teu,
                              needs_deep_berth=needs_deep.astype(int),
                              service_h=service_h, berth_start_h=berth_start,
                              departure_h=departure))
    world = dict(wind=wind, strikes=strikes, strike_hour=strike_hour, maint=maint,
                 queue_len=queue_len, busy_berths=busy_berths)
    return ships, world


# ---------------------------------------------------------------------------
# Turning the simulation into a dataset: "what was known 48 h before arrival"
# ---------------------------------------------------------------------------
def build_records(rng, ships, world, p):
    H = C.PREDICTION_HORIZON_H
    hours = len(world["wind"])
    ships = ships.copy()
    # announced arrival times (ETAs) are imprecise
    ships["reported_eta_h"] = ships["arrival_h"] + rng.normal(0, C.ETA_ERROR_SD_H, len(ships))
    start = pd.Timestamp(C.SIM_START)

    rows = []
    for _, s in ships.iterrows():
        a = int(s.arrival_h)
        t0 = a - H                                    # moment of prediction
        if t0 < C.WARMUP_DAYS * 24 or s.berth_start_h < 0:
            continue    # skip warm-up period and ships still waiting when the simulation ends
        window_end = min(a + 36, hours)

        # --- features: things known at t0 -----------------------------------
        others = ships[(ships.index != s.name)]
        expected_arrivals = int(((others.reported_eta_h >= t0) & (others.reported_eta_h < a)).sum())
        berths_available = p["berths"] - int(world["maint"][a])
        # a 2-day-ahead forecast = the true wind + a forecast error
        forecast = world["wind"][t0:window_end] + rng.normal(0, C.WIND_FORECAST_ERROR_SD) \
            + rng.normal(0, 2, window_end - t0)
        forecast_wind = max(0.0, forecast.max())
        forecast_storm_h = int((forecast >= C.STORM_CLOSURE_WIND_KMH).sum())
        strike_notice = int(any(st["announced"] and st["start"] < a + 48 and st["end"] > t0
                                for st in world["strikes"]))
        recent = ships[(ships.berth_start_h >= t0 - 168) & (ships.berth_start_h < t0)]
        recent_wait = (recent.berth_start_h - recent.arrival_h).mean() if len(recent) else np.nan
        ts = start + pd.Timedelta(hours=a)

        # --- outcome: only known afterwards ---------------------------------
        wait_h = int(s.berth_start_h - a)
        storm_hours = int((world["wind"][a:int(s.berth_start_h) + 1] >= C.STORM_CLOSURE_WIND_KMH).sum())
        strike_hours = int(world["strike_hour"][a:int(s.berth_start_h) + 1].sum())

        rows.append(dict(
            port=s.port,
            arrival_time=ts,
            prediction_time=ts - pd.Timedelta(hours=H),
            # features (known 48 h before arrival)
            ships_waiting_now=int(world["queue_len"][t0]),
            berths_busy_now=int(world["busy_berths"][t0]),
            expected_arrivals_48h=expected_arrivals,
            berths_total=p["berths"],
            berths_available=berths_available,
            forecast_max_wind_kmh=round(forecast_wind, 1),
            forecast_storm_hours=forecast_storm_h,
            strike_notice=strike_notice,
            vessel_teu=s.teu,
            needs_deep_berth=s.needs_deep_berth,
            recent_avg_wait_7d_h=round(recent_wait, 1) if pd.notna(recent_wait) else np.nan,
            arrival_hour=ts.hour,
            day_of_week=ts.dayofweek,
            month=ts.month,
            # outcomes (known only afterwards - NEVER used as model inputs)
            wait_hours=wait_h,
            turnaround_hours=int(s.departure_h - a) if s.departure_h >= 0 else np.nan,
            storm_hours_while_waiting=storm_hours,
            strike_hours_while_waiting=strike_hours,
            delayed=int(wait_h > C.DELAY_THRESHOLD_H),
        ))
    return pd.DataFrame(rows)


def inject_data_problems(rng, df):
    """Real data is messy. Blank out some values the way real feeds fail."""
    df = df.copy()
    for col, rate in C.MISSING_RATE.items():
        mask = rng.random(len(df)) < rate
        df.loc[mask, col] = np.nan
        if col == "forecast_max_wind_kmh":      # the whole weather feed is down
            df.loc[mask, "forecast_storm_hours"] = np.nan
    return df


def main():
    import os
    os.makedirs(os.path.dirname(C.DATA_PATH), exist_ok=True)
    rng = np.random.default_rng(C.RANDOM_SEED)
    hours = C.SIM_DAYS * 24
    frames = []
    for name, p in C.PORTS.items():
        ships, world = simulate_port(rng, name, p, hours)
        recs = build_records(rng, ships, world, p)
        frames.append(recs)
        print(f"{name:11s} ships={len(recs):4d}  delayed={recs.delayed.mean():5.1%}  "
              f"median wait={recs.wait_hours.median():5.1f} h  "
              f"storm-closed hours={int((world['wind'] >= C.STORM_CLOSURE_WIND_KMH).sum())}  "
              f"strikes={len(world['strikes'])}")
    df = pd.concat(frames).sort_values("arrival_time").reset_index(drop=True)
    df = inject_data_problems(rng, df)
    df.insert(0, "call_id", [f"PC{i:05d}" for i in range(len(df))])
    df.to_csv(C.DATA_PATH, index=False)
    print(f"\nSaved {len(df)} port calls to {C.DATA_PATH}  "
          f"(overall delay rate {df.delayed.mean():.1%})")


if __name__ == "__main__":
    main()
