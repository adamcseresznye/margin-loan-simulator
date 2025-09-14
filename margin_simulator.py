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
    """Simulates a margin loan scenario over time with realistic features."""
    if random_seed is not None:
        np.random.seed(random_seed)
    months = years * 12
    portfolio_value = np.zeros(months + 1)
    loan_balance = np.zeros(months + 1)
    equity = np.zeros(months + 1)
    cumulative_interest = np.zeros(months + 1)
    margin_call = np.zeros(months + 1, dtype=bool)
    portfolio_value[0] = initial_portfolio
    loan_balance[0] = initial_loan
    equity[0] = portfolio_value[0] - loan_balance[0]
    monthly_interest_rate = annual_interest_rate / 12
    monthly_dividend_yield = annual_dividend_yield / 12
    monthly_volatility = annual_volatility / np.sqrt(12)
    monthly_expected_return = (1 + annual_market_return) ** (1 / 12) - 1
    for month in range(1, months + 1):
        market_return = np.random.normal(
            loc=monthly_expected_return, scale=monthly_volatility
        )
        portfolio_value[month] = portfolio_value[month - 1] * (
            1 + market_return + monthly_dividend_yield
        )
        interest = loan_balance[month - 1] * monthly_interest_rate
        cumulative_interest[month] = cumulative_interest[month - 1] + interest
        if capitalize_interest:
            loan_balance[month] = loan_balance[month - 1] + interest
        else:
            loan_balance[month] = loan_balance[month - 1]
        if month % 12 == 0:
            portfolio_value[month] += yearly_contribution
            portfolio_value[month] = max(0, portfolio_value[month] - yearly_withdrawal)
            max_loan = portfolio_value[month] * max_loan_to_value
            if loan_balance[month] > max_loan:
                sell_amount = loan_balance[month] - max_loan
                portfolio_value[month] -= sell_amount
                loan_balance[month] = max_loan
        equity[month] = portfolio_value[month] - loan_balance[month]
        if equity[month] < 0:
            margin_call[month] = True
            portfolio_value[month] += equity[month]
            equity[month] = 0
            loan_balance[month] = portfolio_value[month]
    df = pd.DataFrame(
        {
            "Month": range(months + 1),
            "Year": [(m + 11) // 12 for m in range(months + 1)],
            "PortfolioValue": portfolio_value,
            "LoanBalance": loan_balance,
            "Equity": equity,
            "CumulativeInterest": cumulative_interest,
            "MarginCall": margin_call,
        }
    )
    return df[df["Month"] % 12 == 0].reset_index(drop=True)


def plot_margin_loan(df):
    """Plots portfolio, loan, equity, cumulative interest, and margin calls."""
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1.plot(
        df.Year,
        df.PortfolioValue,
        label="Portfolio Value",
        color="tab:blue",
        linewidth=2,
    )
    ax1.plot(
        df.Year, df.LoanBalance, label="Loan Balance", color="tab:red", linewidth=2
    )
    ax1.plot(df.Year, df.Equity, label="Equity", color="tab:green", linewidth=2)
    ax1.fill_between(df.Year, df.Equity, 0, color="tab:green", alpha=0.1)
    ax1.set_xlabel("Years", fontsize=12)
    ax1.set_ylabel("Portfolio / Loan / Equity", fontsize=12)
    ax1.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax2 = ax1.twinx()
    ax2.plot(
        df.Year,
        df.CumulativeInterest,
        label="Cumulative Interest",
        color="tab:orange",
        linestyle="--",
        linewidth=2,
    )
    ax2.set_ylabel("Cumulative Interest", fontsize=12)
    ax2.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    if df["MarginCall"].any():
        mc_years = df.loc[df["MarginCall"], "Year"]
        mc_values = df.loc[df["MarginCall"], "PortfolioValue"]
        ax1.scatter(
            mc_years, mc_values, color="red", marker="x", s=100, label="Margin Call"
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
    start_loan = df["LoanBalance"].iloc[0]
    end_loan = df["LoanBalance"].iloc[-1]
    portfolio_growth = ((end_portfolio / start_portfolio) - 1) * 100
    total_interest = df["CumulativeInterest"].iloc[-1]
    margin_calls = df[df["MarginCall"]]
    margin_call_years = margin_calls["Year"].tolist()

    prompt = (
        f"A margin loan simulation over {total_years} years shows: "
        f"portfolio grew from ${start_portfolio:,.0f} to ${end_portfolio:,.0f} "
        f"({portfolio_growth:.1f}% growth), loan increased from ${start_loan:,.0f} "
        f"to ${end_loan:,.0f}, total interest was ${total_interest:,.0f}. "
    )

    if margin_call_years:
        prompt += f"Margin calls occurred in years {margin_call_years}. "
    else:
        prompt += "No margin calls occurred. "

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
    start_portfolio = df["PortfolioValue"].iloc[0]
    end_portfolio = df["PortfolioValue"].iloc[-1]
    portfolio_growth = ((end_portfolio / start_portfolio) - 1) * 100
    margin_calls = df[df["MarginCall"]]

    if portfolio_growth > 200:
        performance = "excellent"
    elif portfolio_growth > 100:
        performance = "strong"
    else:
        performance = "moderate"

    risk_level = "low risk" if margin_calls.empty else "higher risk due to margin calls"

    return (
        f"they achieved {performance} portfolio growth over {total_years} years with {risk_level}. "
        f"This demonstrates {'successful' if portfolio_growth > 100 else 'cautious'} use of leverage "
        f"for amplifying market returns."
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

    plot_margin_loan(df)

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
