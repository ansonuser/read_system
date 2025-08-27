from transformers import pipeline

pipe = pipeline("text-classification", model="qhchina/BERT-JFJB-violence-0.1")


story = """
  Sure, let's see.\n\n- requirement: financial literacy, market awareness, risk tolerance, regulatory compliance, data access\n\n**Step\u202f1: Build a Solid Foundation**  \n- Study the fundamentals of economics: supply & demand, macro indicators, and fiscal policy.  \n- Read reputable books and take online courses on investing, trading, and market psychology.  \n- Subscribe to high\u2011quality research portals (Reuters, Bloomberg, FactSet).\n\n**Step\u202f2: Understand Market Mechanics**  \n- Learn how exchanges operate (NYSE, NASDAQ, CME, crypto hubs).  \n- Examine order types (market, limit, stop, iceberg) and liquidity concepts.  \n- Simulate trades on paper or via a demo account to grasp execution dynamics.\n\n**Step\u202f3: Perform Technical & Fundamental Analysis**  \n- Technical: master chart patterns, moving averages, oscillators, and volume profiles.  \n- Fundamental: analyze financial statements, earnings reports, competitive positioning, and industry trends.  \n- Combine both to identify high\u2011conviction setups.\n\n**Step\u202f4: Craft a Robust Strategy**  \n- Decide on your trading style: day trade, swing, position, systematic algo, or passive.  \n- Define entry/exit rules, position sizing, and risk\u2011reward ratios (e.g., 2:1 or better).  \n- Backtest over multi\u2011cycle data, validate performance metrics (Sharpe, Sortino, max drawdown).\n\n**Step\u202f5: Develop a Risk Management Framework**  \n- Set a maximum drawdown limit per month (e.g., 5\u201110\u202f% of capital).  \n- Use stop\u2011losses, trailing stops, or volatility\u2011adjusted sizing.  \n- Maintain a diversified portfolio to mitigate idiosyncratic risk.\n\n**Step\u202f6: Operational Readiness**  \n- Choose a broker with low latency, reliable APIs, and comprehensive tools.  \n- Automate monitoring with alerts for adverse market moves or liquidity gaps.  \n- Ensure compliance with regulations (know\u2011your\u2011customer, reporting, and tax filings).\n\n**Step\u202f7: Execute and Iterate**  \n- Execute trades according to your plan, strictly following your time\u2011boxed rules.  \n- Keep a detailed trading journal logging rationale, outcomes, emotions, and lessons learned.  \n- Review performance monthly; refine the strategy based on objective data, not hindsight bias.\n\n**Step\u202f8: Scale with Caution**  \n- Gradually increase position sizes as confidence and consistency grow.  \n- Reinforce risk controls to maintain relative risk exposure.  \n- Rebalance the portfolio periodically to align with revised market forecasts.\n\n**Step\u202f9: Continuous Learning & Adaptation**  \n- Stay updated on macro news, regulatory changes, and emerging asset classes.  \n- Engage in communities, attend webinars, and practice advanced techniques (options, futures).  \n- Keep refining models and incorporating new data (alternatives, ESG metrics).\n\n**Step\u202f10: Psychological Mastery**  \n- Manage stress with breathing exercises, scheduled breaks, and a healthy work\u2013life balance.  \n- Avoid over\u2011trading and emotional decision\u2011making by sticking to pre\u2011established plan.  \n- Periodically re\u2011evaluate your investment goals and risk appetite to stay aligned.\n\nBy rigorously following these steps, you\u2019ll systematically \u201ccrush\u201d the financial market through disciplined, informed, and adaptive trading practices.
"""

n = len(story)
print("n =", n)
chunk_size = 512
slide = 30

for i in range(0, n, chunk_size - slide):
    chunk = story[i:i + chunk_size]
    res = pipe(chunk)
    print(res)
    # if res[0]["label"] == "nsfw" and res[0]["score"] > 0.5:
    #     print("NSFW content detected:")
    #     print(chunk)
    #     break 