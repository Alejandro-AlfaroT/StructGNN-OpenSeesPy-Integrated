import numpy as np
import matplotlib.pyplot as plt

# Given values
cost_total = 7.98       # $ million
bid_total = 8.40        # $ million
retainage_rate = 0.10
duration_weeks = 36
payment_delay = 2
pay_interval = 2

# Lawrence-Miller cumulative percent curve
def lm_percent(x):
    """
    x = percentage time, from 0 to 100
    returns y = percentage value/cost complete
    """
    x = np.asarray(x)
    y = np.zeros_like(x, dtype=float)

    phase1 = x <= 33.33
    phase2 = (x > 33.33) & (x <= 66.67)
    phase3 = x > 66.67

    y[phase1] = 0.0225 * x[phase1]**2
    y[phase2] = 1.5 * x[phase2] - 25
    y[phase3] = 100 - 0.0225 * (100 - x[phase3])**2

    return y

# Time array extends to week 38 because final payment is delayed
t = np.linspace(0, duration_weeks + payment_delay, 2000)

# Cost curve stops increasing after week 36
t_for_cost = np.minimum(t, duration_weeks)
x_percent_time = (t_for_cost / duration_weeks) * 100
y_percent_cost = lm_percent(x_percent_time)
cum_cost = cost_total * (y_percent_cost / 100)

# Payment curve
n_periods = duration_weeks // pay_interval
gross_bill = bid_total / n_periods
net_payment = gross_bill * (1 - retainage_rate)
retainage_total = bid_total * retainage_rate

payment_weeks = [i * pay_interval + payment_delay for i in range(1, n_periods + 1)]
payments = [net_payment] * n_periods
payments[-1] += retainage_total

cum_received = np.zeros_like(t)
for pw, pmt in zip(payment_weeks, payments):
    cum_received += np.where(t >= pw, pmt, 0)

# Find max negative cash flow just before payment dates
candidate_weeks = list(range(0, duration_weeks + 1, 2))
neg_cash_flows = []

for w in candidate_weeks:
    x = (w / duration_weeks) * 100
    c = cost_total * (lm_percent(np.array([x]))[0] / 100)
    received_before = sum(p for pw, p in zip(payment_weeks, payments) if pw < w)
    neg_cash_flows.append(c - received_before)

max_neg = max(neg_cash_flows)
max_week = candidate_weeks[neg_cash_flows.index(max_neg)]
received_before_max = sum(p for pw, p in zip(payment_weeks, payments) if pw < max_week)

# Plot
plt.figure(figsize=(10, 5.8))
plt.plot(t, cum_cost, linewidth=2, label="Cumulative cost (Lawrence-Miller S-curve)")
plt.step(t, cum_received, where="post", linewidth=2, label="Cumulative received payments")

plt.axvline(36, linestyle="--", linewidth=1, label="Project completion, week 36")
plt.axvline(max_week, linestyle=":", linewidth=1.5)
plt.scatter([max_week], [received_before_max], zorder=5)

plt.annotate(
    f"Max negative cash flow\n${max_neg:.2f}M near week {max_week}",
    xy=(max_week, received_before_max),
    xytext=(18, 5.8),
    arrowprops=dict(arrowstyle="->"),
)

plt.xlim(0, 38)
plt.xticks(list(range(0, 39, 2)))
plt.xlabel("Week")
plt.ylabel("Cumulative amount ($ million)")
plt.title("Cost and Received Payment Curves Using Lawrence-Miller S-Curve")
plt.grid(True, alpha=0.35)
plt.legend()
plt.tight_layout()

path = "lawrence_miller_cash_flow_curve.png"
plt.savefig(path, dpi=200)
plt.show()

print(f"Max negative cash flow = ${max_neg:.3f} million near week {max_week}")
print(f"Plot saved to: {path}")