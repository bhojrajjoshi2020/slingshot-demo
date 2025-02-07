import pandas as pd

data = {
    "Employee": ["Alice", "Bob", "Charlie", "David"],
    "Salary": [70000, 80000, 65000, 90000],
    "Department": ["HR", "Engineering", "Finance", "Engineering"]
}

df = pd.DataFrame(data)
average_salary = df.groupby("Department")["Salary"].mean()

print("Average Salary by Department:")
print(average_salary)


average_salary.to_csv("salary_analysis.csv", index=True)
