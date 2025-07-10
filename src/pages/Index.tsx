import HeroSection from "@/components/HeroSection";
import DashboardCard from "@/components/DashboardCard";
import CodeBlock from "@/components/CodeBlock";
import InsightCard from "@/components/InsightCard";
import { Badge } from "@/components/ui/badge";

const Index = () => {
  const clusterCode = `import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load your Excel file
df = pd.read_excel("gaming_survey.xlsx")

# Select only 4 key columns
selected_columns = [
    "Age",
    "How often do you play video games?",
    "Which device do you play games on the most?(Check all that apply)",
    "Why do you play video games? (Check all that apply)"
]
df_selected = df[selected_columns].copy()

# Fill missing values
for col in df_selected.select_dtypes(include="number"):
    df_selected[col] = df_selected[col].fillna(df_selected[col].mean())
for col in df_selected.select_dtypes(include="object"):
    df_selected[col] = df_selected[col].fillna(df_selected[col].mode()[0])

# One-hot encode categorical columns
df_encoded = pd.get_dummies(df_selected, drop_first=True)

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_encoded)

# Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# Apply PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# Plot the clusters
plt.figure(figsize=(8, 6))
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=clusters, cmap='tab10', s=50)
plt.title("Simplified Clustering of Gaming Survey")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label='Cluster')
plt.grid(True)
plt.tight_layout()
plt.savefig("simplified_clusters.png")
plt.show()`;

  const genreCode = `import pandas as pd
import matplotlib.pyplot as plt

# Load the Excel file
df = pd.read_excel("gaming_survey.xlsx")

# Relevant columns
genre_col = "What genres of video games do you play? (Check all that apply)"
freq_col = "How often do you play video games?"

# Drop rows with missing values in either column
df_filtered = df[[genre_col, freq_col]].dropna()

# Explode the genres (in case of multiple selected)
df_filtered = df_filtered.assign(
    Genre=df_filtered[genre_col].str.split(',')
).explode('Genre')
df_filtered['Genre'] = df_filtered['Genre'].str.strip()
df_filtered[freq_col] = df_filtered[freq_col].str.strip()

# Group and pivot the data
genre_freq = df_filtered.groupby(['Genre', freq_col]).size().unstack(fill_value=0)

# Plot the grouped bar chart
genre_freq.plot(kind='bar', figsize=(12, 6))
plt.title("Game Genres by Gaming Frequency", fontsize=14)
plt.xlabel("Game Genre", fontsize=12)
plt.ylabel("Number of Participants", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.legend(title="Gaming Frequency")
plt.tight_layout()
plt.savefig("genre_by_frequency_barchart.png")
plt.show()`;

  const predictionCode = `import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Load the dataset
df = pd.read_csv('gaming_survey.xlsx - Sheet1.csv')

# Select the first 13 columns and clean column names
df = df.iloc[:, :13]
df.columns = df.columns.str.strip().str.replace('[^A-Za-z0-9_]+', '', regex=True)

# Drop 'Timestamp' and 'Whatisyourfavoritegame' as they are not useful features
df = df.drop(columns=['Timestamp', 'Whatisyourfavoritegame'])

# Define ordinal mappings for relevant columns
how_often_mapping = {
    'Rarely/Never': 0,
    'A few times a month': 1,
    'A few times a week': 2,
    'Daily': 3
}

how_many_hours_mapping = {
    '0': 0,
    'Less than 5 hours': 1,
    '5-10 hours': 2,
    '10-20 hours': 3,
    'More than 20 hours': 4
}

monthly_spend_mapping = {
    'Less than ₹100': 0,
    '₹100-500': 1,
    '₹500-1000': 2,
    'More than ₹1000': 3
}

# Apply ordinal mapping
df['Howoftendoyouplayvideogames'] = df['Howoftendoyouplayvideogames'].map(how_often_mapping)
df['Howmanyhoursdoyoutypicallyspendgaminginaweek'] = df['Howmanyhoursdoyoutypicallyspendgaminginaweek'].map(how_many_hours_mapping)
df['Howmuchdoyouspendongamingmonthlyincludingingamepurchasesnewgamesetc']
 = df['Howmuchdoyouspendongamingmonthlyincludingingamepurchasesnewgamesetc'].map(monthly_spend_mapping)

# Clean 'Gender' and 'Location' columns by stripping spaces and converting to lowercase
df['Gender'] = df['Gender'].str.strip().str.lower()
df['Location'] = df['Location'].str.strip().str.lower()

# Multi-label binarization for multi-select columns
multi_select_cols = [
    'WhichdevicedoyouplaygamesonthemostCheckallthatapply',
    'WhatgenresofvideogamesdoyouplayCheckallthatapply',
    'HowdoyoudiscovernewgamesCheckallthatapply',
    'WhydoyouplayvideogamesCheckallthatapply'
]

for col in multi_select_cols:
    df[col] = df[col].astype(str).apply(lambda x: [item.strip() for item in x.split(',')])
    mlb = MultiLabelBinarizer()
    encoded_data = mlb.fit_transform(df[col])
    encoded_df = pd.DataFrame(encoded_data, columns=[f"{col}_{c}" for c in mlb.classes_])
    df = pd.concat([df, encoded_df], axis=1)
    df = df.drop(columns=[col])

# Define the target variable 'y' and features 'X'
target_column = 'Doyouprefersingleplayerormultiplayergames'
y = df[target_column]
X = df.drop(columns=[target_column])

# One-hot encode nominal features within X
nominal_cols_in_X = [
    'Gender',
    'Location'
]

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), nominal_cols_in_X)
    ],
    remainder='passthrough'
)

# Apply transformations to X
X_encoded = preprocessor.fit_transform(X)

# Get feature names after one-hot encoding
ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(nominal_cols_in_X)
remaining_cols = [col for col in X.columns if col not in nominal_cols_in_X]
all_feature_names = np.concatenate([ohe_feature_names, remaining_cols])

# Convert X_encoded back to a DataFrame with proper column names
X_encoded = pd.DataFrame(X_encoded, columns=all_feature_names)

# Handle missing values in numerical/ordinal features (fill with mode)
for col in ['Howoftendoyouplayvideogames', 'Howmanyhoursdoyoutypicallyspendgaminginaweek',
 'Howmuchdoyouspendongamingmonthlyincludingingamepurchasesnewgamesetc']:
    if col in X_encoded.columns:
        X_encoded[col].fillna(X_encoded[col].mode()[0], inplace=True)

# Encode the target variable (y)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.3, random_state=42)

# Initialize and train the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Calculate accuracy and print it with additional text as a percentage
accuracy = accuracy_score(y_test, y_pred)
print(f"The prediction accuracy is: {accuracy:.2%}")`;

const preprocessingCode = `import pandas as pd

# Load the gaming survey file
survey_df = pd.read_excel("gaming_survey (1).xlsx")

# Define the set of columns to be dropped explicitly
columns_to_drop = {
    'S. No.', 'Birthplace', 'Birthplace_ISO3', 'Residence', 'Residence_ISO3', 'Degree', 'Work', 'earnings', 'streams',
    'Game', 'Platform', 'Hours', 'Playstyle', 'Reference', 'whyplay', 'highestleague', 'League', 'Narcissism', 'accept',
    'GADE', 'GAD_T', 'SPIN_T', 'SWL_T'
}

# Add range-based column names (GAD1–GAD7, SPIN1–SPIN17, SWL1–SWL5)
columns_to_drop.update([f'GAD{i}' for i in range(1, 8)])
columns_to_drop.update([f'SPIN{i}' for i in range(1, 18)])
columns_to_drop.update([f'SWL{i}' for i in range(1, 6)])

# Drop the specified columns, ignoring errors for any missing ones
survey_df_cleaned = survey_df.drop(columns=columns_to_drop, errors='ignore')

# Keep only the first 500 rows
survey_df_cleaned = survey_df_cleaned.head(500)

# Save as 'gaming_survey.xlsx'
survey_df_cleaned.to_excel("gaming_survey.xlsx", index=False)

print("File saved as 'gaming_survey.xlsx'")
`;

  return (
    <div className="min-h-screen bg-background">
      <HeroSection />
      
      {/* Problem Statement Section */}
      <section className="py-16 px-6">
        <div className="max-w-6xl mx-auto">
          <DashboardCard 
            title="Problem Statement" 
            icon="🎯"
            glowEffect={true}
            className="mb-8"
          >
            <div className="text-center py-8">
              <p className="text-2xl font-semibold text-primary mb-4">
                "Can we predict player spending and game preference from the survey data?"
              </p>
            </div>
          </DashboardCard>
        </div>
      </section>
 <Badge className="w-fit mx-auto flex justify-center mt-2 mb-6 pb-2 pt-2 bg-green-500 hover:bg-green-500 hover:text-inherit hover:shadow-none hover:border-transparent">
  <h2 className="text-2xl font-semibold text-center bg-gradient-to-r from-gray-300 via-white to-gray-300 bg-clip-text text-transparent">
    Phase-1 Data Pre-Processing
  </h2>
</Badge>



     {/*pre processing */}
     <div className="max-w-6xl mx-auto">
                {/* Code */}
          <div className="mb-8">
            <CodeBlock
              title="Data Pre-processing"
              code={preprocessingCode}/>
          </div>
<div className="p-6 rounded-lg bg-primary/10 border border-primary/20 mt-8 transition duration-300 hover:bg-primary/20 hover:scale-[1.02] hover:shadow-md">
  <p className="font-semibold text-primary mb-4 text-lg">Dataset Preprocessing Summary</p>

  <div className="grid grid-cols-2 gap-6 text-muted-foreground mb-6">
    <ul className="space-y-1 list-disc list-inside">
      <li>Loads the gaming survey Excel file</li>
      <li>Drops unnecessary demographic and psychological columns</li>
      <li>Adds GAD, SPIN, SWL columns using range generation</li>
      <li>Cleans the dataset by dropping irrelevant fields</li>
    </ul>
    <ul className="space-y-1 list-disc list-inside">
      <li>Keeps only the first 500 rows for analysis</li>
      <li>Outputs the cleaned dataset to a new Excel file</li>
      <li>Ensures no errors occur if some columns are missing</li>
      <li>Provides a compact version of the original survey</li>
    </ul>
  </div>
  <div className="flex justify-center">
  <a href="/gaming_survey.xlsx" download>
    <button className="bg-primary text-white font-medium py-2 px-4 rounded hover:bg-primary/90 transition">
      📥 Download Dataset
    </button>
    </a>
    </div>
    </div>     
     </div>
<Badge className="w-fit mx-auto flex justify-center mt-6 pb-2 pt-2 bg-green-500 hover:bg-green-500 hover:text-inherit hover:shadow-none hover:border-transparent">
  <h2 className="text-2xl font-semibold text-center bg-gradient-to-r from-gray-300 via-white to-gray-300 bg-clip-text text-transparent">
    Phase-2 Data Representation
  </h2>
</Badge>

      {/* Cluster Analysis Section */}
      <section className="py-16 px-6 bg-gradient-hero">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-4xl font-bold text-center mb-12 bg-gradient-primary bg-clip-text text-transparent">
            Cluster Analysis: Player Segmentation
          </h2>
          
         {/* Code */}
          <div className="mb-8">
            <CodeBlock
              title="🐍 Python: K-Means Clustering Analysis"
              code={clusterCode}
            />
          </div>
                    <DashboardCard title="Cluster Visualization" icon="📊" glowEffect={true} className="mb-8">
            <div className="bg-muted/20 rounded-lg p-8 border border-border/30">
              <div className="aspect-video bg-gradient-to-br from-primary/20 to-accent/20 rounded-lg flex items-center justify-center border border-primary/20">
                <div className="text-center">

  <p className="text-lg text-muted-foreground mb-2">Cluster Analysis Visualization</p>
  <div className="flex justify-center">
    <img 
      src="/clusters.png"
      alt="Cluster Visualization Graph"
      className="rounded-lg shadow-md w-full max-w-2xl"/>
            </div>
             </div>
              </div>
              <div className="grid grid-cols-3 gap-6 mt-8">
                <div className="text-center p-4 rounded-lg bg-primary/10 border border-primary/20">
                  <div className="w-6 h-6 bg-primary rounded-full mx-auto mb-3"></div>
                  <p className="font-semibold text-primary">Cluster 0</p>
                  <p className="text-sm text-muted-foreground">Young Casuals</p>
                </div>
                <div className="text-center p-4 rounded-lg bg-accent/10 border border-accent/20">
                  <div className="w-6 h-6 bg-accent rounded-full mx-auto mb-3"></div>
                  <p className="font-semibold text-accent">Cluster 1</p>
                  <p className="text-sm text-muted-foreground">Core Gamers</p>
                </div>
                <div className="text-center p-4 rounded-lg bg-secondary/10 border border-secondary/20">
                  <div className="w-6 h-6 bg-secondary rounded-full mx-auto mb-3"></div>
                  <p className="font-semibold text-secondary">Cluster 2</p>
                  <p className="text-sm text-muted-foreground">Enthusiasts</p>
                </div>
              </div>
            </div>
          </DashboardCard>

          {/* Information */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <InsightCard 
              title="Cluster 0: Young Casuals" 
              icon="🎯"
              type="primary"
            >
              <ul className="space-y-2">
                <li><strong>Profile:</strong> Younger participants who play casually</li>
                <li><strong>Platform:</strong> Primarily mobile devices</li>
                <li><strong>Motivation:</strong> Fun and occasional relaxation</li>
                <li><strong>Strategy:</strong> Target with casual, mobile-optimized games</li>
              </ul>
            </InsightCard>

            <InsightCard 
              title="Cluster 1: Core Gamers" 
              icon="⚡"
              type="accent"
            >
              <ul className="space-y-2">
                <li><strong>Profile:</strong> Mixed age, frequent/daily players</li>
                <li><strong>Platform:</strong> Multiple platforms (PC + Mobile)</li>
                <li><strong>Motivation:</strong> Fun, stress relief, improvement</li>
                <li><strong>Strategy:</strong> Regular content drops and engagement loops</li>
              </ul>
            </InsightCard>

            <InsightCard 
              title="Cluster 2: Enthusiasts" 
              icon="🚀"
              type="success">
              <ul className="space-y-2">
                <li><strong>Profile:</strong> Niche players with specific habits</li>
                <li><strong>Platform:</strong> PC or consoles preferred</li>
                <li><strong>Motivation:</strong> Social, nostalgia, long-term goals</li>
                <li><strong>Strategy:</strong> Community-driven content</li>
              </ul>
            </InsightCard>
          </div>
        </div>
      </section>

      {/* Genre Analysis Section */}
      <section className="py-16 px-6">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-4xl font-bold text-center mb-12 bg-gradient-accent bg-clip-text text-transparent">
            Genre Preference Analysis
          </h2>
          {/* Code */}
          <div className="mb-8">
            <CodeBlock
              title="📊 Python: Genre Preference Visualization"
              code={genreCode}
            />
          </div>
                    {/* Graph */}
          <DashboardCard title="Genre by Frequency Chart" icon="📈" glowEffect={true} className="mb-8">
            <div className="bg-muted/20 rounded-lg p-8 border border-border/30">
              <div className="aspect-video bg-gradient-to-br from-accent/20 to-secondary/20 rounded-lg flex items-center justify-center border border-accent/20">
                <div className="text-center">
                  <p className="text-lg text-muted-foreground mb-2">Genre by Frequency Analysis</p>
                   <div className="flex justify-center">
                   <img 
                    src="/bar.png"
                    alt="Cluster Visualization Graph"
                    className="rounded-lg shadow-md w-full max-w-4xl"/>
                 </div>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-8 mt-8">
                <div className="p-4 rounded-lg bg-primary/10 border border-primary/20">
                  <p className="font-semibold text-primary mb-3">Daily Players Prefer:</p>
                  <ul className="text-muted-foreground space-y-1">
                    <li>• FPS (First Person Shooter)</li>
                    <li>• Action/Adventure Games</li>
                    <li>• Horror & Thriller</li>
                  </ul>
                </div>
                <div className="p-4 rounded-lg bg-accent/10 border border-accent/20">
                  <p className="font-semibold text-accent mb-3">Casual Players Prefer:</p>
                  <ul className="text-muted-foreground space-y-1">
                    <li>• Puzzle/Strategy Games</li>
                    <li>• Simulation Games</li>
                    <li>• Sports Games</li>
                  </ul>
                </div>
                 <div className="p-4 rounded-lg bg-accent/10 border border-accent/20">
                  <p className="font-semibold text-accent mb-3">Weekly Players Prefer:</p>
                  <ul className="text-muted-foreground space-y-1">
                    <li>• Action/Adventure Games</li>
                    <li>• Simulation Games</li>
                    <li>• Role-Playing Games</li>
                  </ul>
                </div>
              </div>
            </div>
          </DashboardCard>

          {/* Information */}
          <InsightCard 
            title="Genre Preferences by Gaming Frequency - Key Findings" 
            icon="🎮"
            type="warning"
          >
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div>
                <h4 className="font-semibold mb-4 text-card-foreground text-lg">High-Engagement Players</h4>
                <ul className="space-y-2">
                  <li><strong>FPS Games:</strong> Dominate among daily players - fast-paced, competitive gameplay appeals to frequent gamers</li>
                  <li><strong>Action/Adventure:</strong> Strong preference for narrative-driven, immersive experiences</li>
                  <li><strong>Horror Games:</strong> Adrenaline-seeking behavior correlates with higher gaming frequency</li>
                </ul>
              </div>
              <div>
                <h4 className="font-semibold mb-4 text-card-foreground text-lg">Casual Gaming Preferences</h4>
                <ul className="space-y-2">
                  <li><strong>Puzzle/Strategy:</strong> Most popular among infrequent players - thoughtful, low-pressure gameplay</li>
                  <li><strong>Simulation Games:</strong> Universal appeal across all frequency levels - accessible and relaxing</li>
                  <li><strong>Sports Games:</strong> Consistent engagement regardless of gaming frequency</li>
                </ul>
              </div>
              <div>
  <h4 className="font-semibold mb-4 text-card-foreground text-lg">Weekly Player Preferences</h4>
  <ul className="space-y-2">
    <li><strong>Action/Adventure:</strong> Weekly players favor immersive storytelling and balanced gameplay intensity</li>
    <li><strong>Simulation Games:</strong> Offers relaxing, sandbox-style experiences suited for regular but not daily sessions</li>
    <li><strong>Role-Playing Games:</strong> Preferred for their depth and character progression without the need for daily play</li>
  </ul>
</div>

            </div>
          </InsightCard>
        </div>
      </section>

      {/* Machine Learning Model Section */}
      <section className="py-16 px-6 bg-gradient-hero">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-4xl font-bold text-center mb-12 bg-gradient-primary bg-clip-text text-transparent">
            Random Forest Classification Model
          </h2>
          
          {/* Code */}
          <div className="mb-8">
            <CodeBlock
              title="🤖 Python: Random Forest Classification Model"
              code={predictionCode}
            />
          </div>
          {/* Graph */}
          <DashboardCard 
  title="Model Performance Visualization" 
  icon="🤖"
  glowEffect={true}
  className="mb-8">
  <div className="bg-muted/20 rounded-lg p-4 border border-border/30">
  <div className="flex flex-col md:flex-row gap-6 items-center justify-between">
    {/* Animated Accuracy Card */}
<div className="aspect-[4/2] w-full max-w-md rounded-md flex items-center justify-center border border-secondary/20 relative overflow-hidden bg-[length:400%_400%] bg-gradient-to-r from-blue-900 via-purple-600 to-indigo-800 motion-safe:animate-[gradient_6s_ease-in-out_infinite]">
  <div className="absolute inset-0 bg-gradient-to-r from-indigo-900 via-violet-500 to-blue-800 bg-[length:400%_400%] animate-gradient opacity-60"></div>


      <div className="text-center z-10">
<div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-primary rounded-full mb-4 shadow-[inset_0_2px_4px_rgba(255,255,255,0.2),0_4px_8px_rgba(0,0,0,0.25)] border border-white/10 backdrop-blur-sm">
    <span className="text-xl font-bold text-primary-foreground drop-shadow-sm">82%</span>
    </div>
         <p className="text-base font-semibold bg-gradient-to-r from-gray-200 via-white to-gray-400 bg-clip-text text-transparent mb-1">
           Prediction Accuracy Achieved
        </p>
        <p className="text-xs font-medium bg-gradient-to-r from-gray-300 via-gray-100 to-gray-500 bg-clip-text text-transparent">
         Random Forest Classifier Performance
        </p>
      </div>

      <div className="absolute top-3 right-3 flex gap-2 z-10">
        <div className="w-2.5 h-2.5 bg-success rounded-full animate-pulse"></div>
        <div className="w-2.5 h-2.5 bg-primary rounded-full animate-pulse" style={{ animationDelay: '0.2s' }}></div>
        <div className="w-2.5 h-2.5 bg-accent rounded-full animate-pulse" style={{ animationDelay: '0.4s' }}></div>
      </div>
    </div>

    {/* Metrics Cards aligned to right */}
    <div className="grid grid-cols-1 md:grid-cols-1 gap-4 w-full md:w-1/2">
      <div className="text-center p-4 rounded-lg bg-success/10 border border-success/20">
        <div className="w-5 h-5 bg-success rounded-full mx-auto mb-2"></div>
        <p className="font-medium text-success text-sm">High Accuracy</p>
        <p className="text-xs text-muted-foreground">82% Success Rate</p>
      </div>
      <div className="text-center p-4 rounded-lg bg-primary/10 border border-primary/20">
        <div className="w-5 h-5 bg-primary rounded-full mx-auto mb-2"></div>
        <p className="font-medium text-primary text-sm">Feature Correlation</p>
        <p className="text-xs text-muted-foreground">Strong Predictive Power</p>
      </div>
      <div className="text-center p-4 rounded-lg bg-accent/10 border border-accent/20">
        <div className="w-5 h-5 bg-accent rounded-full mx-auto mb-2"></div>
        <p className="font-medium text-accent text-sm">Model Robustness</p>
        <p className="text-xs text-muted-foreground">Reliable Predictions</p>
      </div>
    </div>
  </div>
</div>
</DashboardCard>
          {/* Information */}
          <InsightCard 
            title="Random Forest Classifier: Results & Analysis" 
            icon="🔬"
            type="success"
          >
            <div className="space-y-6">
              <div>
                <h4 className="font-semibold mb-3 text-card-foreground text-lg">Objective</h4>
                <p className="text-muted-foreground leading-relaxed">
                  The Random Forest Classifier was utilized to predict an individual's preference for single-player 
                  or multiplayer games based on various survey factors including age, gender, location, gaming frequency, 
                  hours spent gaming, preferred gaming devices, game genres, game discovery methods, and monthly gaming expenditure.
                </p>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div>
                  <h4 className="font-semibold mb-3 text-card-foreground text-lg">Key Results</h4>
                  <ul className="space-y-2 text-muted-foreground">
                    <li><strong className="text-success">82% Accuracy:</strong> Model correctly predicted gaming preferences for most test subjects</li>
                    <li><strong className="text-primary">Feature Importance:</strong> Strong correlation between player profiles and game type preferences</li>
                    <li><strong className="text-accent">Model Reliability:</strong> Consistent performance across different test scenarios</li>
                  </ul>
                </div>
                
                <div>
                  <h4 className="font-semibold mb-3 text-card-foreground text-lg">Business Impact</h4>
                  <ul className="space-y-2 text-muted-foreground">
                    <li><strong className="text-primary">Player Targeting:</strong> Identify optimal game types for different user segments</li>
                    <li><strong className="text-accent">Content Strategy:</strong> Data-driven decisions for game development priorities</li>
                    <li><strong className="text-success">Revenue Optimization:</strong> Improved user engagement through personalized recommendations</li>
                  </ul>
                </div>
              </div>
            </div>
          </InsightCard>
        </div>
      </section>

      {/* Footer */}
<footer className="bg-gradient-to-tr from-white to-slate-100 border-t border-slate-200 shadow-inner">
  <div className="max-w-7xl mx-auto px-6 py-2 flex flex-col md:flex-row items-center justify-between text-xs md:text-sm text-slate-600 space-y-3 md:space-y-0 md:space-x-8">
    
    {/* Left - Branding */}
    <div className="flex items-center gap-2 text-slate-700">
      <span className="font-semibold tracking-wide text-gray-800">© 2025 Chandan Sathvik</span>
      <span className="hidden md:inline-block w-1 h-1 bg-slate-400 rounded-full"></span>
      <span className="italic">All rights reserved</span>
    </div>

    {/* Center - Student ID + Role */}
    <div className="flex items-center gap-2 text-slate-600">
      <span className="text-blue-600 font-bold">ID:</span>
      <span className="font-mono text-gray-800">21MIC7040</span>
    </div>
  </div>
</footer>
    </div>
  );
};

export default Index;