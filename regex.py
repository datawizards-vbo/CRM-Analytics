import pandas as pd

# Create a sample DataFrame
data = {
    'id': [1, 2, 3, 4, 5],
    'name': ['John Doe', 'Jane Smith 123', 'Alice Brown', 'Bob Johnson', 'Charlie Lee 42'],
    'age': [25, 30, 45, 50, 35],
    'email': ['johndoe@gmail.com', 'jane.smith@yahoo.com', 'alice_brown@outlook.com', 'bob.johnson@gmail.com', 'charlie_lee@yahoo.com']
}
df = pd.DataFrame(data)

# Example 1: Extract the last word from the name column
df['last_name'] = df['name'].str.extract(r'(\w+)$')
print(df)

# Example 2: Remove all whitespace characters from the email column
df['email'] = df['email'].str.replace(r'\s', '')
print(df)

# Example 3: Find rows that contain 'yahoo.com' or 'outlook.com' in the email column
df_yahoo_outlook = df[df['email'].str.contains(r'yahoo\.com|outlook\.com')]
print(df_yahoo_outlook)

# Example 4: Find rows where the name column starts with 'J' and ends with 'e'
df_je_names = df[df['name'].str.contains(r'^J.*e$')]
print(df_je_names)

# Example 5: Replace all vowels in the name column with 'X'
df['name'] = df['name'].str.replace(r'[aeiou]', 'X', regex=True)
print(df)

# Example 6: Find rows that contain a number in the name column
df_with_numbers = df[df['name'].str.contains(r'\d')]
print(df_with_numbers)

# Example 7: Replace all non-word characters in the name column with a space
df['name'] = df['name'].str.replace(r'\W', ' ', regex=True)
print(df)

# Example 8: Find rows where the name column contains two words separated by a space
df_two_word_names = df[df['name'].str.contains(r'\w+\s\w+')]
print(df_two_word_names)



