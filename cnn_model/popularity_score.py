from datetime import datetime
import numpy as np
from sklearn.preprocessing import PowerTransformer
from joblib import dump
from tqdm import tqdm
import os

views_weight = 0.34
likes_weight = 0.33
comment_count_weight =  0.33

def days_since(start_date, end_date=None):
    start_date = datetime.strptime(start_date, "%Y-%m-%dT%H:%M:%SZ")
    if end_date is None:
        time_difference = datetime.utcnow() - start_date
    else:
        end_date = datetime.strptime(end_date, "%Y-%m-%dT%H:%M:%SZ")
        time_difference = end_date - start_date
    return time_difference.days

def z_score(video_feature, feature):
    # Subtract the mean from each data point divided by stddev and store in a list
    return (video_feature - feature.mean()) / feature.std()

def generate_pop_score(video, df):
    return (
        z_score(video.number_of_views, df['number_of_views']) * views_weight + \
        z_score(video.number_of_likes, df['number_of_likes']) * likes_weight + \
        z_score(video.number_of_comments, df['number_of_comments']) * comment_count_weight
    )

def generate_pop_scores(filename, df):
    # check to see if we can load mfccs from file
    if os.path.exists(filename):
        pop_scores = np.load(filename)
    else:
        # create an array of all video likes
        pop_scores = np.array([generate_pop_score(video, df) for video in \
                        tqdm(df.itertuples(), total=len(df), desc=f"Generating Popularity Scores for {filename}")])
        pop_scores = pop_scores.reshape(-1, 1)
        pt = PowerTransformer(method='yeo-johnson')
        pop_scores = pt.fit_transform(pop_scores)
        pop_scores = np.ravel(pop_scores)
        # save the fitted transformer
        dump(pt, f'{filename.split(".")[0]}_fitted_transformer.joblib')
        np.save(filename, pop_scores)
    return pop_scores

# df = pd.read_csv("../youtube_shorts.csv")
# df = df[~(df['publication_date_time'].apply(lambda x: days_since(x)) < 7)]
# df = df[~(df['subscriber_count'] < 10000)]
# print(df[~(df['number_of_views'] > 10000)])

# for col_name, col_series in df.items():
#     if np.issubdtype(col_series.dtype, np.number):
#         col_series = col_series.apply(lambda x: np.log1p(x))
#         if col_name == 'number_of_comments':
#             col_series = col_series.apply(lambda x: np.cbrt(x))
#         print(f"Mean of {col_name}: {col_series.mean()}")
#         print(f"Standard Deviation of {col_name}: {col_series.std()}")
#         print(f"Variance of {col_name}: {col_series.var()}")
#         print()

#         # Create a new figure and axes for each plot
#         fig, ax = plt.subplots()

#         # Generate your seaborn plot on the specific axes
#         sns.histplot(data=col_series, ax=ax)
#         ax.set(xlabel=col_name, ylabel='Frequency', title=f'Distribution of {col_name}')

#         # Save the plot as a PNG file
#         fig.savefig(f'plots/hist_plot_{col_name}.png', dpi=300)  # Specify the filename and desired DPI (dots per inch)

#         # Close the figure to free up resources
#         plt.close(fig)