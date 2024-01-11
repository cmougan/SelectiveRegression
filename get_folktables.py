from folktables import (
    ACSDataSource,
    ACSIncome,
    ACSPublicCoverage,
    ACSMobility,
    BasicProblem,
    generate_categories,
)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import sklearn.metrics as skm
import os

from sklearn.preprocessing import MinMaxScaler

us_states = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
    "District of Columbia": "DC",
}
state_list = [
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
    "PR",
]

ACSIncome_categories = {
    "COW": {
        1.0: (
            "Employee of a private for-profit company or"
            "business, or of an individual, for wages,"
            "salary, or commissions"
        ),
        2.0: (
            "Employee of a private not-for-profit, tax-exempt,"
            "or charitable organization"
        ),
        3.0: "Local government employee (city, county, etc.)",
        4.0: "State government employee",
        5.0: "Federal government employee",
        6.0: (
            "Self-employed in own not incorporated business,"
            "professional practice, or farm"
        ),
        7.0: (
            "Self-employed in own incorporated business,"
            "professional practice or farm"
        ),
        8.0: "Working without pay in family business or farm",
        9.0: "Unemployed and last worked 5 years ago or earlier or never worked",
    },
    "SCHL": {
        1.0: "No schooling completed",
        2.0: "Nursery school, preschool",
        3.0: "Kindergarten",
        4.0: "Grade 1",
        5.0: "Grade 2",
        6.0: "Grade 3",
        7.0: "Grade 4",
        8.0: "Grade 5",
        9.0: "Grade 6",
        10.0: "Grade 7",
        11.0: "Grade 8",
        12.0: "Grade 9",
        13.0: "Grade 10",
        14.0: "Grade 11",
        15.0: "12th grade - no diploma",
        16.0: "Regular high school diploma",
        17.0: "GED or alternative credential",
        18.0: "Some college, but less than 1 year",
        19.0: "1 or more years of college credit, no degree",
        20.0: "Associate's degree",
        21.0: "Bachelor's degree",
        22.0: "Master's degree",
        23.0: "Professional degree beyond a bachelor's degree",
        24.0: "Doctorate degree",
    },
    "MAR": {
        1.0: "Married",
        2.0: "Widowed",
        3.0: "Divorced",
        4.0: "Separated",
        5.0: "Never married or under 15 years old",
    },
    "SEX": {1.0: "Male", 2.0: "Female"},
    "RAC1P": {
        1.0: "White alone",
        2.0: "Black or African American alone",
        3.0: "American Indian alone",
        4.0: "Alaska Native alone",
        5.0: (
            "American Indian and Alaska Native tribes specified;"
            "or American Indian or Alaska Native,"
            "not specified and no other"
        ),
        6.0: "Asian alone",
        7.0: "Native Hawaiian and Other Pacific Islander alone",
        8.0: "Some Other Race alone",
        9.0: "Two or More Races",
    },
}


def adult_filter(data):
    """Mimic the filters in place for Adult data.

    Adult documentation notes: Extraction was done by Barry Becker from
    the 1994 Census database. A set of reasonably clean records was extracted
    using the following conditions:
    ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))
    """
    df = data
    df = df[df["AGEP"] > 16]
    df = df[df["PINCP"] > 100]
    df = df[df["WKHP"] > 0]
    df = df[df["PWGTP"] >= 1]
    return df


ACSIncomeImb = BasicProblem(
    features=[
        "AGEP",
        "COW",
        "SCHL",
        "MAR",
        "OCCP",
        "POBP",
        "RELP",
        "WKHP",
        "SEX",
        "RAC1P",
    ],
    target="PINCP",
    target_transform=lambda x: x > 75000,
    group="RAC1P",
    preprocess=adult_filter,
    postprocess=lambda x: np.nan_to_num(x, -1),
)

ACSIncomeReg = BasicProblem(
    features=[
        'AGEP',
        'COW',
        'SCHL',
        'MAR',
        'OCCP',
        'POBP',
        'RELP',
        'WKHP',
        'SEX',
        'RAC1P',
    ],
    target='PINCP',
    group='RAC1P',
    preprocess=adult_filter,
    postprocess=lambda x: np.nan_to_num(x, -1),
)


data_source = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
bsize = {}
states_to_check = ["MN", "ND", "NV", "OK", "UT", "WV"]
for state in state_list:
    print(state)
    ca_data = data_source.get_data(states=[state], download=True)
    ca_features, ca_labels, _ = ACSIncomeReg.df_to_pandas(
        ca_data, categories=ACSIncome_categories, dummies=False
    )
    print(np.mean(ca_labels))
    print(ca_labels.shape[0])

    # / crea un dataframe pandas

    ca_features.columns = [
        "_".join(i).replace(",", "") for i in ca_features.columns.str.split(" ")
    ]
    df = ca_features.copy()
    df["TARGET"] = ca_labels.astype(int)
    df["AGEP"] = df["AGEP"].astype(int)
    # df["PINCP"] = df["PINCP"].astype(float)
    df["TARGET"] = ca_labels.copy()
    # df.drop("PINCP", axis=1, inplace=True)
    atts = list(ca_features.columns)
    cat_atts = list(df[atts].select_dtypes(include=["object", "category"]).columns)
    for col in cat_atts:
        df[col] = col + df[col].astype(str)
    # / split dataset in four datasets: train, test, holdout and calibration
    # we use a different seed for "ACSPublicCoverage2018MN","ACSPublicCoverage2018ND", # "ACSPublicCoverage2018NV",
    # "ACSPublicCoverage2018OK", "ACSPublicCoverage2018UT" and "ACSPublicCoverage2018WV" as the random split with 42
    # has issues in the categorical features distribution and it is not balanced across the splits, creating issues with
    # the training of the models
    if os.path.exists("data/folktables") is False:
        os.mkdir("data/folktables")
    df.to_csv(
        "data/folktables/FOLKTABLES_US_2018_INCOME_{}.csv".format(
            state, state
        ),
        index=False,
    )
print(atts)
