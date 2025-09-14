import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import requests


def simulate_margin_loan(
    initial_portfolio=100000,
    initial_loan=50000,
    annual_market_return=0.08,
    annual_volatility=0.15,
    annual_dividend_yield=0.02,
    annual_interest_rate=0.07,
    years=30,
    yearly_contribution=0,
    yearly_withdrawal=0,
    capitalize_interest=True,
    max_loan_to_value=0.5,
    random_seed=None,
):
    """
    Simulates a margin loan scenario over time with a robust financial model.

    Args:
        initial_portfolio (float): The starting value of the portfolio.
        initial_loan (float): The initial amount of the margin loan.
        annual_market_return (float): The expected annual price return (ex-dividends).
        annual_volatility (float): The annual volatility of portfolio returns.
        annual_dividend_yield (float): The annual dividend yield.
        annual_interest_rate (float): The annual interest rate on the margin loan.
        years (int): The number of years to simulate.
        yearly_contribution (float): Annual contribution to the portfolio.
        yearly_withdrawal (float): Annual withdrawal from the portfolio.
        capitalize_interest (bool): If True, interest is added to the loan balance.
                                    If False, assumes interest is paid from an external
                                    cash source and does not affect the portfolio value.
        max_loan_to_value (float): The maximum allowed loan-to-value ratio.
        random_seed (int, optional): Seed for the random number generator.

    Returns:
        pd.DataFrame: A DataFrame with the yearly simulation results.
    """
    rng = np.random.default_rng(random_seed)
    months = years * 12
    portfolio_value = np.zeros(months + 1)
    loan_balance = np.zeros(months + 1)
    equity = np.zeros(months + 1)
    cumulative_interest = np.zeros(months + 1)
    margin_call = np.zeros(months + 1, dtype=bool)
    insolvent = np.zeros(months + 1, dtype=bool)

    portfolio_value[0] = initial_portfolio
    loan_balance[0] = initial_loan
    equity[0] = portfolio_value[0] - loan_balance[0]

    monthly_interest_rate = (1 + annual_interest_rate) ** (1 / 12) - 1
    monthly_dividend_yield = annual_dividend_yield / 12
    monthly_sigma = annual_volatility / np.sqrt(12)
    monthly_drift = np.log(1 + annual_market_return) / 12 - 0.5 * monthly_sigma**2

    for month in range(1, months + 1):
        z = rng.normal()
        monthly_price_return = np.exp(monthly_drift + monthly_sigma * z) - 1
        total_multiplier = (1 + monthly_price_return) * (1 + monthly_dividend_yield) - 1
        portfolio_value[month] = portfolio_value[month - 1] * (1 + total_multiplier)

        interest = loan_balance[month - 1] * monthly_interest_rate
        cumulative_interest[month] = cumulative_interest[month - 1] + interest
        loan_balance[month] = loan_balance[month - 1] + (
            interest if capitalize_interest else 0
        )

        if month % 12 == 0:
            portfolio_value[month] += yearly_contribution
            portfolio_value[month] = max(0, portfolio_value[month] - yearly_withdrawal)

        max_loan = portfolio_value[month] * max_loan_to_value
        if loan_balance[month] > max_loan:
            margin_call[month] = True
            s_required = (loan_balance[month] - max_loan) / (1 - max_loan_to_value)
            s_required = max(0.0, s_required)

            if s_required >= portfolio_value[month]:
                proceeds = portfolio_value[month]
                portfolio_value[month] = 0.0
                loan_balance[month] = max(0.0, loan_balance[month] - proceeds)
                if loan_balance[month] > 0:
                    insolvent[month] = True
            else:
                portfolio_value[month] -= s_required
                loan_balance[month] -= s_required

        equity[month] = portfolio_value[month] - loan_balance[month]

    df = pd.DataFrame(
        {
            "Month": range(months + 1),
            "Year": [m // 12 for m in range(months + 1)],
            "PortfolioValue": portfolio_value,
            "LoanBalance": loan_balance,
            "Equity": equity,
            "CumulativeInterest": cumulative_interest,
            "MarginCall": margin_call,
            "Insolvent": insolvent,
        }
    )
    return df[df["Month"] % 12 == 0].reset_index(drop=True)


def plot_margin_loan(df, max_loan_to_value):
    """Plots portfolio, loan, equity, and margin calls, including the LTV threshold."""
    fig, ax1 = plt.subplots(figsize=(12, 7))
    df["MaintenanceMargin"] = df.PortfolioValue * max_loan_to_value

    ax1.plot(
        df.Year,
        df.PortfolioValue,
        label="Portfolio Value",
        color="tab:blue",
        linewidth=2,
    )
    ax1.plot(df.Year, df.Equity, label="Equity", color="tab:green", linewidth=2)
    ax1.fill_between(df.Year, df.Equity, 0, color="tab:green", alpha=0.1)
    ax1.set_xlabel("Years", fontsize=12)
    ax1.set_ylabel("Portfolio / Equity", fontsize=12)
    ax1.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax1.axhline(0, color="grey", linewidth=0.5, linestyle="--")

    ax2 = ax1.twinx()
    ax2.plot(
        df.Year, df.LoanBalance, label="Loan Balance", color="tab:red", linewidth=2
    )
    ax2.plot(
        df.Year,
        df.CumulativeInterest,
        label="Cumulative Interest",
        color="tab:orange",
        linestyle="--",
        linewidth=2,
    )

    ax2.set_ylabel("Loan / Interest", fontsize=12)
    ax2.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))

    if df["MarginCall"].any():
        mc_years = df.loc[df["MarginCall"], "Year"]
        mc_values = df.loc[df["MarginCall"], "PortfolioValue"]
        ax1.scatter(
            mc_years,
            mc_values,
            color="orange",
            marker="X",
            s=100,
            label="Margin Call",
            zorder=5,
        )

    if df["Insolvent"].any():
        insolvent_years = df.loc[df["Insolvent"], "Year"]
        ax1.scatter(
            insolvent_years,
            [0] * len(insolvent_years),
            color="red",
            marker="o",
            s=100,
            label="Insolvency",
            zorder=5,
        )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2, labels1 + labels2, frameon=False, fontsize=10, loc="upper left"
    )
    ax1.set_title("Margin Loan Simulation", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.show()


def df_to_text_prompt(df):
    total_years = df["Year"].max()
    start_portfolio = df["PortfolioValue"].iloc[0]
    end_portfolio = df["PortfolioValue"].iloc[-1]
    start_equity = df["Equity"].iloc[0]
    end_equity = df["Equity"].iloc[-1]

    portfolio_cagr_str = "N/A"
    if start_portfolio > 0 and total_years > 0:
        portfolio_cagr = (end_portfolio / start_portfolio) ** (1.0 / total_years) - 1
        portfolio_cagr_str = f"{portfolio_cagr:.2%}"

    equity_cagr_str = "N/A"
    if start_equity > 0 and total_years > 0:
        equity_cagr = (end_equity / start_equity) ** (1.0 / total_years) - 1
        equity_cagr_str = f"{equity_cagr:.2%}"

    total_interest = df["CumulativeInterest"].iloc[-1]
    margin_call_years = df[df["MarginCall"]]["Year"].unique().tolist()
    insolvency_years = df[df["Insolvent"]]["Year"].unique().tolist()

    prompt = (
        f"Over {total_years} years, the portfolio grew from ${start_portfolio:,.0f} to ${end_portfolio:,.0f} "
        f"(Portfolio CAGR: {portfolio_cagr_str}). The investor's equity grew from ${start_equity:,.0f} to ${end_equity:,.0f} "
        f"(Equity CAGR: {equity_cagr_str}). Total interest was ${total_interest:,.0f}. "
    )

    if margin_call_years:
        prompt += f"Margin calls occurred in years: {margin_call_years}. "
    if insolvency_years:
        prompt += f"Insolvency (portfolio wiped out with debt remaining) occurred in years: {insolvency_years}. "
    if not margin_call_years and not insolvency_years:
        prompt += "No margin calls or insolvency events occurred."

    return prompt


def get_ai_explanation_api(prompt_text, hf_token):
    """Generate AI explanation using Hugging Face API with reliable models."""
    if not hf_token or hf_token.strip() == "":
        return "⚠️ Please enter your Hugging Face API token to get AI explanations."

    models_to_try = ["gpt2", "distilgpt2", "facebook/bart-large-cnn"]

    for model in models_to_try:
        try:
            API_URL = f"https://api-inference.huggingface.co/models/{model}"
            headers = {"Authorization": f"Bearer {hf_token}"}

            if "bart" in model:
                payload = {
                    "inputs": prompt_text,
                    "parameters": {
                        "max_length": 150,
                        "min_length": 40,
                        "do_sample": True,
                        "temperature": 0.7,
                    },
                }
            else:
                payload = {
                    "inputs": prompt_text,
                    "parameters": {
                        "max_new_tokens": 80,
                        "temperature": 0.7,
                        "repetition_penalty": 1.2,
                        "top_p": 0.9,
                        "do_sample": True,
                        "return_full_text": False,
                    },
                }

            response = requests.post(API_URL, headers=headers, json=payload, timeout=30)

            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    if "bart" in model:
                        explanation = result[0].get("summary_text", "")
                    else:
                        generated = result[0].get("generated_text", "")
                        if generated.startswith(prompt_text):
                            explanation = generated[len(prompt_text) :].strip()
                        else:
                            explanation = generated

                    if explanation.strip():
                        return explanation.strip()

            elif response.status_code == 401:
                return "❌ Invalid Hugging Face token. Please check your token and try again."
            elif response.status_code == 503:
                continue
            elif response.status_code == 404:
                continue

        except Exception as e:
            continue

    return (
        "🔄 All models are currently unavailable. This can happen when:\n"
        "• Models are loading (wait 1-2 minutes and try again)\n"
        "• High API traffic\n"
        "• Temporary Hugging Face API issues\n"
        "Try again in a few minutes."
    )


def get_basic_explanation(df):
    """Fallback explanation when AI is unavailable."""
    total_years = df["Year"].max()
    start_equity = df["Equity"].iloc[0]
    end_equity = df["Equity"].iloc[-1]

    equity_growth = 0
    if start_equity > 0:
        equity_growth = ((end_equity / start_equity) - 1) * 100

    if df["Insolvent"].any():
        performance = "a catastrophic failure, ending in insolvency"
    elif equity_growth > 200:
        performance = "excellent"
    elif equity_growth > 100:
        performance = "strong"
    else:
        performance = "moderate"

    risk_level = "low risk"
    if df["Insolvent"].any():
        risk_level = "extreme risk, leading to a total loss"
    elif df["MarginCall"].any():
        risk_level = "higher risk due to margin calls"

    return (
        f"they achieved {performance} equity growth over {total_years} years with {risk_level}. "
        f"This demonstrates {'the successful' if equity_growth > 100 else 'the cautious'} use of leverage. "
    )


def run_simulation_and_analysis(
    initial_portfolio,
    initial_loan,
    annual_market_return,
    annual_volatility,
    annual_dividend_yield,
    annual_interest_rate,
    years,
    yearly_contribution,
    yearly_withdrawal,
    capitalize_interest,
    max_loan_to_value,
    hf_token,
):
    """Main function that runs simulation and provides analysis."""
    df = simulate_margin_loan(
        initial_portfolio=initial_portfolio,
        initial_loan=initial_loan,
        annual_market_return=annual_market_return,
        annual_volatility=annual_volatility,
        annual_dividend_yield=annual_dividend_yield,
        annual_interest_rate=annual_interest_rate,
        years=years,
        yearly_contribution=yearly_contribution,
        yearly_withdrawal=yearly_withdrawal,
        capitalize_interest=capitalize_interest,
        max_loan_to_value=max_loan_to_value,
        random_seed=42,
    )

    plot_margin_loan(df, max_loan_to_value)

    prompt = df_to_text_prompt(df)
    print("\n" + "=" * 60)
    print("AI EXPLANATION OF RESULTS")
    print("=" * 60)

    if hf_token and hf_token.strip():
        explanation = get_ai_explanation_api(prompt, hf_token)
        print(explanation)

        if (
            "unavailable" in explanation
            or "Error" in explanation
            or "🔄" in explanation
        ):
            print("\n" + "-" * 40)
            print("BASIC EXPLANATION (Fallback)")
            print("-" * 40)
            basic_explanation = get_basic_explanation(df)
            print(f"For an investor, this means {basic_explanation}")
    else:
        basic_explanation = get_basic_explanation(df)
        print(f"For an investor, this means {basic_explanation}")
        print(
            "\n💡 Enter your Hugging Face token above to get AI-powered explanations!"
        )
