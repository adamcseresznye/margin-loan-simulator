import ipywidgets as widgets
from ipywidgets import VBox, HBox, Layout

# Description HTML widget
description = widgets.HTML(
    """
<div style="background: #ffffff; padding: 20px; border-radius: 10px; margin: 20px 0; border: 2px solid #e5e7eb; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
<h3 style="color: #1f2937;">📈 Interactive Margin Loan Simulator</h3>
<p style="color: #4b5563;">Explore the evolution of a leveraged portfolio using margin loans. Adjust parameters to see how portfolio growth, interest costs, and margin calls affect your investment strategy.</p>
<p style="color: #4b5563;"><strong>💡 Tip:</strong> Get a free Hugging Face token from <a href="https://huggingface.co/settings/tokens" target="_blank" style="color: #2563eb;">huggingface.co/settings/tokens</a> for AI-powered explanations!</p>
</div>
"""
)

# Create all widgets with detailed tooltips
portfolio_slider = widgets.IntSlider(
    value=100000,
    min=50000,
    max=500000,
    step=10000,
    description="Portfolio",
    tooltip="Initial value of your investment portfolio (e.g., stocks, ETFs). This is the total value of assets you own before borrowing.",
)

loan_slider = widgets.IntSlider(
    value=50000,
    min=10000,
    max=300000,
    step=10000,
    description="Loan",
    tooltip="Amount of money borrowed on margin from your broker. This adds leverage to amplify both gains and losses.",
)

market_slider = widgets.FloatSlider(
    value=0.08,
    min=-0.1,
    max=0.2,
    step=0.01,
    description="Market Return",
    tooltip="Expected average annual return of your portfolio (as decimal: 0.08 = 8%). Historical stock market average is around 7-10%.",
)

volatility_slider = widgets.FloatSlider(
    value=0.15,
    min=0.0,
    max=0.5,
    step=0.01,
    description="Volatility",
    tooltip="Annual fluctuation/risk of your portfolio returns (0.15 = 15%). Higher values mean more unpredictable year-to-year performance.",
)

dividend_slider = widgets.FloatSlider(
    value=0.02,
    min=0.0,
    max=0.1,
    step=0.005,
    description="Dividend Yield",
    tooltip="Annual dividend income as fraction of portfolio value (0.02 = 2%). Many ETFs and dividend stocks pay 1-4% annually.",
)

interest_slider = widgets.FloatSlider(
    value=0.07,
    min=0.0,
    max=0.2,
    step=0.005,
    description="Loan Rate",
    tooltip="Annual interest rate charged on your margin loan (0.07 = 7%). Brokers typically charge 5-10% depending on rates and account size.",
)

years_slider = widgets.IntSlider(
    value=30,
    min=5,
    max=50,
    step=1,
    description="Years",
    tooltip="Number of years to simulate the margin loan strategy. Longer periods show compound effects but add uncertainty.",
)

contribution_slider = widgets.IntSlider(
    value=0,
    min=0,
    max=50000,
    step=1000,
    description="Contribution",
    tooltip="Additional money you add to your portfolio each year. Regular contributions can reduce risk through dollar-cost averaging.",
)

withdrawal_slider = widgets.IntSlider(
    value=0,
    min=0,
    max=20000,
    step=1000,
    description="Withdrawal",
    tooltip="Amount you withdraw annually for expenses. This reduces your portfolio value and available equity each year.",
)

capitalize_checkbox = widgets.Checkbox(
    value=True,
    description="Capitalize Interest",
    tooltip="If checked, unpaid interest is added to your loan balance (compound interest). If unchecked, you pay interest annually in cash.",
)

capitalize_explanation = widgets.HTML(
    """
    <div style="font-size:11px; color:#4b5563; margin-top:-10px; margin-bottom:10px; margin-left:110px; line-height: 1.4;">
        <b>If checked:</b> The loan balance grows as interest is added to it.<br>
        <b>If unchecked:</b> Assumes you pay the monthly interest with cash.
    </div>
    """
)

max_ltv_slider = widgets.FloatSlider(
    value=0.5,
    min=0.0,
    max=1.0,
    step=0.05,
    description="Max LTV",
    tooltip="Maximum Loan-to-Value ratio (0.5 = 50%). When your loan exceeds this percentage of portfolio value, assets are automatically sold to reduce leverage.",
)

hf_token_input = widgets.Password(
    value="",
    description="HF Token:",
    placeholder="Optional: For AI explanations",
    tooltip="Your free Hugging Face API token for AI-powered explanations. Get one at huggingface.co/settings/tokens - completely optional but adds intelligent analysis.",
)

# Create layout
col1 = VBox(
    [portfolio_slider, loan_slider, market_slider, volatility_slider, dividend_slider]
)
col2 = VBox(
    [
        interest_slider,
        years_slider,
        contribution_slider,
        withdrawal_slider,
        capitalize_checkbox,
        capitalize_explanation,
        max_ltv_slider,
    ]
)
col3 = VBox([hf_token_input])
controls = HBox([col1, col2, col3], layout=Layout(justify_content="center"))

# Define what can be imported
__all__ = [
    "description",
    "controls",
    "portfolio_slider",
    "loan_slider",
    "market_slider",
    "volatility_slider",
    "dividend_slider",
    "interest_slider",
    "years_slider",
    "contribution_slider",
    "withdrawal_slider",
    "capitalize_checkbox",
    "max_ltv_slider",
    "hf_token_input",
]
