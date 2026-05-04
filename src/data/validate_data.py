import great_expectations as gx
from typing import Tuple, List


def validate_data(df) -> Tuple[bool, List[str]]:
    print("🔍 Starting data validation with Great Expectations...")
    context = gx.get_context(mode="ephemeral")
    datasource = context.data_sources.add_pandas("pandas_datasource")
    asset = datasource.add_dataframe_asset(name="churn_data")
    batch_definition = asset.add_batch_definition_whole_dataframe("batch_def")
    expectations = gx.ExpectationSuite(name="churn_suite")

    print("   📋 Validating schema and required columns...")
    for col in ["gender", "Partner", "Dependents", "PhoneService",
                "InternetService", "Contract", "tenure",
                "MonthlyCharges", "TotalCharges", "Churn"]:
        expectations.add_expectation(gx.expectations.ExpectColumnToExist(column=col))

    print("   💼 Validating business logic constraints...")
    expectations.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(
        column="gender", value_set=["Male", "Female"]))

    for col in ["Partner", "Dependents", "PhoneService"]:
        expectations.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(
            column=col, value_set=["Yes", "No"]))

    expectations.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(
        column="Contract", value_set=["Month-to-month", "One year", "Two year"]))
    expectations.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(
        column="InternetService", value_set=["DSL", "Fiber optic", "No"]))
    expectations.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(
        column="Churn", value_set=[0, 1]))

    print("   📊 Validating numeric ranges and business constraints...")
    expectations.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(
        column="tenure", min_value=0, max_value=120))
    expectations.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(
        column="MonthlyCharges", min_value=0, max_value=200))
    expectations.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(
        column="TotalCharges", min_value=0, max_value=10000))

    print("   📈 Validating not-null constraints...")
    for col in ["tenure", "MonthlyCharges", "TotalCharges", "Churn"]:
        expectations.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column=col))

    print("   🔗 Validating data consistency...")
    expectations.add_expectation(gx.expectations.ExpectColumnPairValuesAToBeGreaterThanB(
        column_A="TotalCharges",
        column_B="MonthlyCharges",
        or_equal=True,
        mostly=0.95,
    ))

    suite = context.suites.add(expectations)
    validation_definition = context.validation_definitions.add(
        gx.ValidationDefinition(name="churn_validation", data=batch_definition, suite=suite)
    )

    print("   ⚙️  Running complete validation suite...")
    results = validation_definition.run(batch_parameters={"dataframe": df})

    failed_expectations = []
    for r in results.results:
        if not r.success:
            failed_expectations.append(r.expectation_config.type)

    total_checks = len(results.results)
    passed_checks = sum(1 for r in results.results if r.success)
    failed_checks = total_checks - passed_checks

    if results.success:
        print(f"✅ Data validation PASSED: {passed_checks}/{total_checks} checks successful")
    else:
        print(f"❌ Data validation FAILED: {failed_checks}/{total_checks} checks failed")
        for r in results.results:
            if not r.success:
                kwargs = r.expectation_config.kwargs
                col = kwargs.get('column', kwargs.get('column_A', 'N/A'))
                unexpected = r.result.get('unexpected_count', 'N/A')
                print(f"   ❌ {r.expectation_config.type} | col: {col} | unexpected: {unexpected}")

    return results.success, failed_expectations