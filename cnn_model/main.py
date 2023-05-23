import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error
from popularity_score import generate_pop_scores, days_since
from feature_extraction import collect_features
import os
from discord_webhook import DiscordWebhook, DiscordEmbed
import matplotlib.pyplot as plt
import seaborn as sns
import config

def send_discord_message(title, description, color='03b2f8'):
    webhook_url = config.WEBHOOK_URL
    webhook = DiscordWebhook(url=webhook_url)

    # Create an embed
    embed = DiscordEmbed(title=title, description=description, color=color)

    # Add embed to webhook
    webhook.add_embed(embed)

    response = webhook.execute()
    return response

def dict_to_markdown(d):
    markdown = ''
    for key, value in d.items():
        markdown += f'- **{key}**: {value}\n'
    return markdown

send_discord_message("Script Starting", f"", '00ff00')

try:
    if not os.path.exists("data"):
        os.makedirs("data")

    df = pd.read_csv('../youtube_shorts.csv')
    # ensure no None rows in columns we care about
    df = df.dropna(subset=['audio_file', 'number_of_views', 'number_of_likes', 'subscriber_count', \
                        'number_of_comments', 'publication_date_time'])
    # filter out outliers
    df = df[~(df['publication_date_time'].apply(lambda x: days_since(x)) < 7)]
    df = df[~(df['subscriber_count'] < 10000)]

    for col_name, col_series in df.items():
        if np.issubdtype(col_series.dtype, np.number):
            # logarithm transformation
            df[col_name] = col_series.apply(lambda x: np.log1p(x))
            # cube root transformation
            if col_name == 'number_of_comments':
                df[col_name] = col_series.apply(lambda x: np.cbrt(x))

    # Assume df is your DataFrame and you want to predict 'target'
    feature_df = collect_features(df)
    X = feature_df.drop('video_id', axis=1)
    pop_score_df = df[df['video_id'].isin(feature_df['video_id'])]
    y = generate_pop_scores(os.path.join('data', 'pop_scores.npy'), pop_score_df)

    # # Create a new figure and axes for each plot
    # fig, ax = plt.subplots()

    # # Generate your seaborn plot on the specific axes
    # sns.histplot(data=y, ax=ax)
    # ax.set(xlabel='Popularity Score', ylabel='Frequency', title=f'Distribution of Popularity Scores')

    # # Save the plot as a PNG file
    # fig.savefig(f'plots/hist_plot_pop_score.png', dpi=300)  # Specify the filename and desired DPI (dots per inch)

    # # Close the figure to free up resources
    # plt.close(fig)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the grid of hyperparameters to search
    hyperparameter_grid = {
        'alpha': [0, 0.2, 0.4, 0.6, 0.8, 1], 
        'colsample_bytree': [0.5, 0.7, 0.9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'max_depth': [3, 5, 7, 9],
        'n_estimators': [100, 300, 500, 700, 900]
    }

    # Instantiate an XGBRegressor
    xg_reg = xgb.XGBRegressor(objective='reg:squarederror')

    # Set up the grid search with 3-fold cross validation
    grid_cv = GridSearchCV(estimator=xg_reg, param_grid=hyperparameter_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)

    grid_cv.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_cv.best_params_

    # Fit the model with best parameters
    xg_reg_best = xgb.XGBRegressor(objective='reg:squarederror', **best_params)
    xg_reg_best.fit(X_train, y_train)

    preds = xg_reg_best.predict(X_test)

    # Compute the mean absolute error of the predictions
    mae = mean_absolute_error(y_test, preds)

    # Get the best parameters
    best_params = grid_cv.best_params_

    # Compute the mean absolute error of the predictions
    mae = mean_absolute_error(y_test, preds)

    # Save model to file
    xg_reg_best.save_model("model_best.json")

    # Construct message
    msg = '\n'.join([
        '**Hyperparameter grid:**',
        dict_to_markdown(hyperparameter_grid),
        '\n**Best parameters:**',
        dict_to_markdown(best_params),
        f'\n**Mean Absolute Error:**\n{mae}',
    ])

    # Send the message
    send_discord_message("Model Training Results", msg, '00ff00')

except Exception as e:
    # Send error notification
    send_discord_message("Script Failed", f"<@{259910798434893834}>Script has encountered an error: {str(e)}", 'ff0000')