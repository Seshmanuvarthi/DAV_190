import numpy as np
import pandas as pd

# Let's start by grabbing our cricket data from the file.
try:
    df = pd.read_csv('cricket_data_2025.csv')  # Replace with your file name
except FileNotFoundError:
    print("Hey, I couldn't find 'cricket_data_2025.csv'. Make sure it's in the same folder as this code!")
    exit()

# Sometimes, the data has "No stats" instead of blanks. Let's fix that.
df = df.replace("No stats", np.nan)

# We want to make sure all the numbers are actually treated as numbers.
numeric_cols = ['Year', 'Matches_Batted', 'Not_Outs', 'Runs_Scored', 'Balls_Faced',
                'Batting_Average', 'Batting_Strike_Rate', 'Centuries', 'Half_Centuries',
                'Fours', 'Sixes', 'Catches_Taken', 'Stumpings', 'Matches_Bowled',
                'Balls_Bowled', 'Runs_Conceded', 'Wickets_Taken', 'Bowling_Average',
                'Economy_Rate', 'Bowling_Strike_Rate', 'Four_Wicket_Hauls', 'Five_Wicket_Hauls']

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce') # if something is not a number, it will be NaN

# The 'Best_Bowling_Match' column had two stats together. Let's split them.
df[['Best_Bowling_Wickets', 'Best_Bowling_Runs']] = df['Best_Bowling_Match'].str.split('/', expand=True)
df['Best_Bowling_Wickets'] = pd.to_numeric(df['Best_Bowling_Wickets'], errors='coerce')
df['Best_Bowling_Runs'] = pd.to_numeric(df['Best_Bowling_Runs'], errors='coerce')
df = df.drop('Best_Bowling_Match', axis=1) # We don't need the old column anymore.

# We only care about rows with years, so let's drop the others.
df = df.dropna(subset=['Year'])

# If there are any missing values, let's fill them with zeros.
df = df.fillna(0)

# Reset the index, so it's nice and tidy.
df = df.reset_index(drop=True)

print("Here's a sneak peek at our clean data:")
print(df.head())
print('\n')

# --- NumPy: Our Array Helper ---

print("Let's use NumPy to do some number crunching!")

# We'll turn some of our columns into NumPy arrays.
try:
    years = np.array(df['Year'])
    runs_scored = np.array(df['Runs_Scored'])
    wickets_taken = np.array(df['Wickets_Taken'])
except KeyError as e:
    print(f"Oops! I couldn't find the column '{e}'. Double-check the column names!")
    exit()

print("Years:", years[:5])
print("Runs Scored:", runs_scored[:5])
print("Wickets Taken:", wickets_taken[:5])
print("\n")

# Some simple array operations:
print("First Year:", years[0])
print("Third Runs Scored:", runs_scored[2])
print("\n")

print("First 3 Wicket counts:", wickets_taken[:3])
print("Years from index 2 to 5:", years[2:6])
print("\n")

reshaped_runs = runs_scored.reshape((len(runs_scored), 1))
print("Runs Scored as a column:", reshaped_runs[:5])
print("\n")

array_concat = np.concatenate((years[:5], [2025, 2026]))
print("Years with a few extra:", array_concat)

split_index = int(len(years) / 2)
array_split = np.split(years, [split_index])
print("First half of years:", array_split[0][:5])
print("Second half of years:", array_split[1][:5])
print("\n")

squared_runs = np.square(runs_scored)
print("Squared Runs:", squared_runs[:5])
print("\n")

print("Average Runs Scored:", np.mean(runs_scored))
print("Most Wickets Taken:", np.max(wickets_taken))
print("\n")

print("Runs Scored plus 10:", runs_scored[:5] + 10)
print("\n")

print("Wickets over 15?:", wickets_taken[:10] > 15)
print("\n")

print("Years with many wickets:", years[wickets_taken > 15][:10])
print("\n")

print("Runs at specific spots:", runs_scored[[0, 5, 10]])
print("\n")

print("Sorted Runs:", np.sort(runs_scored)[:5])

indices_sorted_by_wickets = np.argsort(wickets_taken)
print("Order of wickets:", indices_sorted_by_wickets[:5])
print("Wickets in order:", wickets_taken[indices_sorted_by_wickets[:5]])
print("\n")

indices_of_smallest_wickets = np.argpartition(wickets_taken, 5)[:5]
print("Indices of 5 lowest wickets:", indices_of_smallest_wickets)
print("5 lowest wickets:", wickets_taken[indices_of_smallest_wickets])
print("\n")

structured_array = np.zeros(5, dtype={'names': ('player_name', 'year'), 'formats': ('U50', 'i4')})
structured_array['player_name'] = df['Player_Name'][:5]
structured_array['year'] = df['Year'][:5]
print("Player names and years together:", structured_array)
print("\n")

record_array = structured_array.view(np.recarray)
print("Just the years from that array:", record_array.year)
print("\n")

# --- Pandas: Our Data Organizer ---

print("Now let's use Pandas to work with our data in a more organized way.")

player_series = pd.Series(df['Player_Name'])
print("First few player names:", player_series.head())
print("\n")

print("DataFrame info:")
df.info()
print("\n")

print("First player name:", player_series[0])
print("Players 2 to 4:", player_series[2:5])
print("\n")

print("Players and Runs:", df[['Player_Name', 'Runs_Scored']].head())
print("\n")
print("Row at index 1:", df.loc[1])
print("\n")

try:
    print("Batting average at index 3:", df.iloc[3, df.columns.get_loc('Batting_Average')])
    print("\n")
except KeyError:
    print("Looks like I couldn't find 'Batting_Average'.")

player_with_prefix = player_series.map(lambda name: "Player: " + name)
print("Players with 'Player:' in front:", player_with_prefix.head())
print("\n")

try:
    run_series = pd.Series(df['Runs_Scored'], index=df['Player_Name'])
    batting_avg_series = pd.Series(df['Batting_Average'], index=df['Player_Name'])
    runs_plus_avg = run_series + batting_avg_series
    print("Runs plus batting averages:", runs_plus_avg.head())
    print("\n")
except KeyError:
    print("Couldn't add runs and averages. Check the column names.")

print("Missing values in each column:")
print(df.isnull().sum())
print("\n")

hierarchical_index = df.set_index(['Player_Name', 'Year'])
print("Data with player names and years as labels:", hierarchical_index.head())
print("\n")

# --- Combining Datasets ---

print("Let's combine some data!")

data2 = {'Player_Name': ['Abdul Samad', 'Abhinav Manohar', 'Adam Zampa'],
         'Favorite Shot': ['Sweep', 'Cover Drive', 'Googly']}
df2 = pd.DataFrame(data2)
print("Favorite shots data:", df2)
print("\n")

concatenated_df = pd.concat([df.head(3), df2], ignore_index=True)
print("Combined data:", concatenated_df)
print("\n")

appended_df = pd.concat([df.head(3), df2], ignore_index=True) #Corrected append to pd.concat
print("Appended data:", appended_df)
print("\n")

merged_df = pd.merge(df, df2, on='Player_Name', how='left')
print("Merged data:", merged_df.head())
print("\n")

try:
    average_runs_by_year = df.groupby('Year')['Runs_Scored'].mean()
    print("Average runs per year:", average_runs_by_year)
    print("\n")
except KeyError:
    print("Couldn't group by year. Check the column name.")

try:
    pivot_table = pd.pivot_table(df, values='Runs_Scored', index='Year', columns='Player_Name', aggfunc='mean')
    print("Runs by year and player:", pivot_table.head())
except KeyError:
    print("Couldn't make the pivot table. Check the column names.")