import logging
import pandas as pd

# Configure logging
logging.basicConfig(format='%(asctime)s %(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Example function to calculate average salary
def calculate_average_salary(dataframe):
    if 'salary' not in dataframe.columns:
        logger.error("The dataframe does not contain a 'salary' column.")
        return None
    average_salary = dataframe['salary'].mean()
    logger.info(f"Calculated average salary: {average_salary}")
    return average_salary

# Example usage
def main():
    # Load data from a CSV file
    try:
        df = pd.read_csv('employee_data.csv')
        logger.info("Data loaded successfully.")
    except FileNotFoundError:
        logger.error("The file 'employee_data.csv' was not found.")
        return

    # Calculate average salary
    avg_salary = calculate_average_salary(df)
    if avg_salary is not None:
        # Save the average salary to a CSV file
        avg_salary_df = pd.DataFrame({'average_salary': [avg_salary]})
        avg_salary_df.to_csv("salary_analysis.csv", index=False)
        logger.info("Average salary saved to 'salary_analysis.csv'.")

if __name__ == "__main__":
    main()
