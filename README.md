import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


# ----------------------------------------------------------------------
# 1. DATA
# ----------------------------------------------------------------------

def generate_synthetic_data(n_houses=500, seed=42):
    """
    Replace with: df = pd.read_csv("houses.csv")

    Expected real columns:
        - plot_area_sqft        (total plot size)
        - built_up_area_sqft    (actual constructed area)
        - floors                (number of floors, integer)
        - bedrooms
        - bathrooms
        - material_quality      (1=basic, 2=standard, 3=premium, 4=luxury)
        - location_tier         (1=rural, 2=town, 3=city, 4=metro)
        - labor_cost_index      (regional labor cost multiplier, ~0.7-1.5)
        - price                 (target: total building cost)
    """
    rng = np.random.default_rng(seed)

    built_up_area = rng.normal(1800, 700, n_houses).clip(400, 6000)
    plot_area = built_up_area * rng.uniform(1.1, 2.0, n_houses)
    floors = rng.integers(1, 4, n_houses)
    bedrooms = rng.integers(1, 6, n_houses)
    bathrooms = rng.integers(1, 5, n_houses)
    material_quality = rng.integers(1, 5, n_houses)   # 1-4
    location_tier = rng.integers(1, 5, n_houses)       # 1-4
    labor_cost_index = rng.normal(1.0, 0.2, n_houses).clip(0.6, 1.6)

    # Base cost per sqft depends on material quality + location + labor
    base_cost_per_sqft = (
        800
        + material_quality * 350
        + location_tier * 200
        + (labor_cost_index - 1) * 500
    )

    price = (
        built_up_area * base_cost_per_sqft
        + floors * 150_000          # extra structural cost per floor
        + bedrooms * 80_000
        + bathrooms * 120_000       # plumbing/fittings are expensive
        + rng.normal(0, 150_000, n_houses)  # noise/market variation
    ).clip(500_000, None)

    df = pd.DataFrame({
        "plot_area_sqft": plot_area,
        "built_up_area_sqft": built_up_area,
        "floors": floors,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "material_quality": material_quality,
        "location_tier": location_tier,
        "labor_cost_index": labor_cost_index,
        "price": price,
    })
    return df


FEATURE_COLUMNS = [
    "plot_area_sqft",
    "built_up_area_sqft",
    "floors",
    "bedrooms",
    "bathrooms",
    "material_quality",
    "location_tier",
    "labor_cost_index",
]


# ----------------------------------------------------------------------
# 2. TRAIN / EVALUATE
# ----------------------------------------------------------------------

def train_model(df, test_size=0.2, seed=42):
    X = df[FEATURE_COLUMNS]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # --- Linear Regression (interpretable baseline) ---
    lin_model = LinearRegression()
    lin_model.fit(X_train_scaled, y_train)
    lin_pred = lin_model.predict(X_test_scaled)
    results["Linear Regression"] = evaluate(y_test, lin_pred)

    print("=== Linear Regression: cost impact per feature (standardized) ===")
    coef_series = pd.Series(lin_model.coef_, index=FEATURE_COLUMNS)
    print(coef_series.sort_values(ascending=False).round(0))
    print()

    # --- Random Forest (stronger, non-linear) ---
    rf_model = RandomForestRegressor(
        n_estimators=300, max_depth=8, min_samples_leaf=3, random_state=seed
    )
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    results["Random Forest"] = evaluate(y_test, rf_pred)

    print("=== Random Forest: feature importance ===")
    imp_series = pd.Series(rf_model.feature_importances_, index=FEATURE_COLUMNS)
    print(imp_series.sort_values(ascending=False).round(3))
    print()

    print("=== Model Comparison ===")
    print(pd.DataFrame(results).T.round(2))

    return lin_model, rf_model, scaler


def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


# ----------------------------------------------------------------------
# 3. PREDICTION HELPER
# ----------------------------------------------------------------------

def predict_price(model, scaler, house_dict):
    """
    house_dict example:
    {
        "plot_area_sqft": 2400,
        "built_up_area_sqft": 1600,
        "floors": 2,
        "bedrooms": 3,
        "bathrooms": 2,
        "material_quality": 3,   # 1-4
        "location_tier": 3,      # 1-4
        "labor_cost_index": 1.0,
    }
    """
    row = pd.DataFrame([house_dict])[FEATURE_COLUMNS]
    row_scaled = scaler.transform(row)
    return model.predict(row_scaled)[0]


# ----------------------------------------------------------------------
# 4. INTERACTIVE USER INPUT
# ----------------------------------------------------------------------

def get_user_input():
    """
    Asks the user for their house's space/details on the command line
    and returns a dict ready for predict_price().
    """
    print("\nEnter your house details to get a predicted building price.")
    print("(Press Enter to accept the default shown in brackets.)\n")

    def ask(prompt, default, cast=float):
        raw = input(f"{prompt} [{default}]: ").strip()
        return cast(raw) if raw else cast(default)

    built_up_area = ask("Built-up (constructed) area in sqft", 1600)
    plot_area = ask("Plot/land area in sqft", built_up_area * 1.5)
    floors = ask("Number of floors", 2, cast=int)
    bedrooms = ask("Number of bedrooms", 3, cast=int)
    bathrooms = ask("Number of bathrooms", 2, cast=int)
    material_quality = ask(
        "Material quality: 1=basic, 2=standard, 3=premium, 4=luxury", 2, cast=int
    )
    location_tier = ask(
        "Location: 1=rural, 2=town, 3=city, 4=metro", 3, cast=int
    )
    labor_cost_index = ask(
        "Local labor cost index (0.7=cheap area, 1.0=average, 1.5=expensive area)",
        1.0,
    )

    return {
        "plot_area_sqft": plot_area,
        "built_up_area_sqft": built_up_area,
        "floors": floors,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "material_quality": material_quality,
        "location_tier": location_tier,
        "labor_cost_index": labor_cost_index,
    }


# ----------------------------------------------------------------------
# 5. MAIN
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating synthetic house dataset (replace with real CSV when ready)...\n")
    df = generate_synthetic_data()

    lin_model, rf_model, scaler = train_model(df)

    joblib.dump(rf_model, "house_price_model.pkl")
    joblib.dump(scaler, "house_price_scaler.pkl")
    print("\nSaved Random Forest model to house_price_model.pkl")

    # --- Interactive prediction for the space the user enters ---
    house = get_user_input()
    predicted_price = predict_price(rf_model, scaler, house)
    print(f"\nPredicted building cost for your {house['built_up_area_sqft']:.0f} sqft house: "
          f"₹{predicted_price:,.0f}")
    print(f"(≈ ₹{predicted_price / house['built_up_area_sqft']:,.0f} per sqft)")
