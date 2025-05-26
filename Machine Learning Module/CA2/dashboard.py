import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
import plotly.express as px
from matplotlib.colors import LinearSegmentedColormap

# Dashboard Link: https://dashboard-ca3.streamlit.app/

# This Dashboard will have two pages
# 1. Analytics
    # All the charts from the Jupyter File for stakeholders
# 2. Recommender System
    # For Users
    # This is a tailored solution for every user. I will run the Euclidean Distance code in the Jupyter file and apply it to this dashboard     
    # This will bring the top 5 recommendations for any user
    # Display the Trailer
    # Filter recommendation by Genres
    # User Analytics
    # Filtering Predictive Rating: The User will be able to get the movie they want by setting a minimum predictive rating desired.

# Streamlit page layout
st.set_page_config(page_title="Movie Recommender", layout="wide")

# Datasets
movies = pd.read_csv('movies.csv', encoding='latin1')
ratings = pd.read_csv('rating.csv', encoding='latin1')
tags = pd.read_csv('tags.csv', encoding='latin1')

# Drop NaNs
movies.dropna(inplace=True)
ratings.dropna(inplace=True)
tags.dropna(inplace=True)

# Merging it
merged = ratings.merge(movies, on='movieId', how='left')

# -------------------------------- Sidebar Navigation -------------------------------- #

# Two pages Naviagtion
page = st.sidebar.selectbox("Choose Section", ["Recommender System", "Analytics"])

# ------------------------------ Recommender System Page ----------------------------- #

if page == "Recommender System":

    # Sidebar Login 
    user_id = st.sidebar.selectbox(
        'Login:', 
        options=[None] + list(ratings['userId'].unique()), 
        index=0, 
        key="login_select"
    )

    # Always run with None selected
    if user_id is None:
        st.sidebar.warning("Please select a user to log in.")
        st.stop() 

#----------------------------- Machine Learning ------------------------------------------# 

    # Euclidean distance from the Jupyter Notebook

    # User-item Pivot Table
    user_movie_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

    # Euclidean distance between users
    euc_dist = euclidean_distances(user_movie_matrix)

    # This avoids comparing user to themselves
    np.fill_diagonal(euc_dist, np.inf) 

    # Distance to Similarity
    euc_sim = 1 / (1 + euc_dist)

    # Pivot Table to df
    euc_sim_df = pd.DataFrame(euc_sim, index=user_movie_matrix.index, columns=user_movie_matrix.index)

    # Recommend movies based on similar users' ratings
    def recommend_movies(user_id, top_n=5):

        # Top 5
        similar_users = euc_sim_df[user_id].sort_values(ascending=False).head(5).index
        top_users_ratings = user_movie_matrix.loc[similar_users]

        # Collaborative Filtering: AVG
        average_ratings = top_users_ratings.mean()

        # Remove watched movies
        user_seen_movies = user_movie_matrix.loc[user_id]
        unrated_movies = average_ratings[user_seen_movies == 0]

        # Sorting
        top_recommendations = unrated_movies.sort_values(ascending=False).head(top_n)
        return top_recommendations

    #--------------------------------- Header ---------------------------------------------# 

    # Lets show Youtube Trailer for the Top predicted Movie (Highest Rating Prediction)
    # NEW (offline version)
    # The API calls for fetching data where changed to a local style, due to the incompatibility with the Streamlit hosting

    trailers_df = pd.read_csv('trailers.csv')
    
    def get_trailer_url(title):
        match = trailers_df[trailers_df['title'] == title]
        if not match.empty:
            return match.iloc[0]['trailer_url']
        return "Trailer not found"

    #######################################################################################################################################################
    # -------------------------------------------------------------- Main Page --------------------------------------------------------------------------#    
    ######################################################################################################################################################

    # ---------------------------- Movies Genre, Trailer and Top Recommendations ----------------#

    st.title(f'Hi {user_id}! You will ❤️ to watch this movie:')

    # Passing the Machine Learning Function
    top_recommendations = recommend_movies(user_id)

    # Top movies Genres filtering recommendations
    recommended_movies = movies[movies['movieId'].isin(top_recommendations.index)]
    top_movies = top_recommendations.reset_index()
    top_movies.columns = ['movieId', 'Predicted Rating']

    # Merging the genre
    top_movies = top_movies.merge(movies[['movieId', 'title', 'genres']], on='movieId')

    # I split all the possible genres and leave only one 
    top_movies['genre_list'] = top_movies['genres'].str.split('|')
    top_movies_exploded = top_movies.explode('genre_list')
    available_genres = sorted(top_movies_exploded['genre_list'].unique())

    # Trailer for top recommendation
    if not top_movies_exploded.empty:

        # Get our movie
        top_movie_title = top_movies_exploded.iloc[0]['title']

        # Get the trailer
        trailer_url = get_trailer_url(top_movie_title)

        # Plotting the trailer
        if trailer_url:
            st.markdown("### 🎞️ " + top_movie_title)
            st.video(trailer_url)
        else:
            st.warning("Trailer not found for this movie.")

    # GENRE SELECTOR
    selected_genre = st.selectbox('🎬 Movies by Genre:', options=['All'] + available_genres)

    # Now to filter movies by selected genre
    filtered_movies = (
        top_movies_exploded[top_movies_exploded['genre_list'] == selected_genre]
        if selected_genre != 'All' 
        else top_movies_exploded
    )
    filtered_movies = filtered_movies.drop_duplicates(subset='movieId').sort_values(by='Predicted Rating', ascending=False)

    # Adding Genre to the Title for the bar chart
    filtered_movies['title_with_genre'] = filtered_movies['title'] + ' | ' + filtered_movies['genre_list'] + ''


    #######################################################################
    # First Section: Horizontal bar chart, taking all the width of the page
    #######################################################################         

    # ---------------------- Horizontal bar chart of Recommended Movies by ML ------------------------- #

    fig_bar = px.bar(
        filtered_movies,
        x='Predicted Rating',
        y='title_with_genre',
        orientation='h',
        text='title_with_genre'
    )
    # Names inside the bar and Bigger Size text
    fig_bar.update_traces(textposition='inside', marker_color='steelblue', textfont_size=20)

    fig_bar.update_layout(

        # To show them in order
        yaxis=dict(autorange="reversed"),

        # Lest get rid off things
        showlegend=False,
        xaxis_visible=False,
        yaxis_visible=False,
        title='🎯 Our Recommendations To You',
        margin=dict(l=20, r=20, t=30, b=20),
        height=400
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    #####################################
    # Second Section: two charts by side
    #####################################    

    st.markdown("### 🎥 Your Movie Preferences Analytics")

    # Splitting the page in two 
    col1, col2 = st.columns([1.5, 1])

    # -------------------------- User rating histogram -------------------- #

    with col1:
        # Get rating per specific user and Count it
        user_ratings = ratings[ratings['userId'] == user_id]
        user_ratings_data = user_ratings['rating'].value_counts().sort_index().reset_index()
        user_ratings_data.columns = ['Rating', 'Count']

        # Plot
        fig_hist = px.bar(user_ratings_data, x='Rating', y='Count', title='⭐ How Much You Like Movies', text='Count')
        fig_hist.update_traces(marker_color='steelblue', textposition='outside')
        fig_hist.update_layout(margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_hist, use_container_width=True)

    # ----------------------------------- Genre pie chart --------------------------#

    with col2:
        # Same as col1 but we get now the top 5 Genres
        genre_counts = filtered_movies['genre_list'].value_counts().head(5).reset_index()
        genre_counts.columns = ['Genre', 'Count']

        # Plot
        fig_pie = px.pie(genre_counts, values='Count', names='Genre', title='Your🔝Genres')

        # Bigger text
        fig_pie.update_traces(textfont_size=20)
        
        st.plotly_chart(fig_pie, use_container_width=True)

    ##########################
    # Third Section: CheckBox
    ##########################

    # --------------------------------- Heatmap of similar users ------------------------ #

    if st.checkbox("👥 See Similar Users Heatmap"):
        st.markdown("#### Heatmap of Your Top 5 Similar Users")

        # Top 5 similar users
        sim_users = euc_sim_df.loc[user_id].sort_values(ascending=False).head(5).index
        sim_matrix = euc_sim_df.loc[sim_users, sim_users]

        # Custom blues scales including our themed 'steelblue'
        custom_blues = LinearSegmentedColormap.from_list(
            'custom_steelblue',
            ['#e0f7fa', 'steelblue', '#08306b']
        )

        # Black background
        fig, ax = plt.subplots()
        fig.patch.set_facecolor('black')       
        ax.set_facecolor('black')           
        
        # Title in white
        ax.set_title("Top 5 Similar Users Heatmap (Darker = More Similar)", color='white') 
        sns.heatmap(sim_matrix, annot=True, cmap=custom_blues, ax=ax, 
                    cbar=True, linewidths=0.5, linecolor='gray')

        # Axis tick labels white
        ax.tick_params(colors='white', which='both')  
        plt.xticks(color='white')
        plt.yticks(color='white')

        st.pyplot(fig)


    # ------------------------ Line Chart: Genre trends over time --------------------------------- #

    # Transforming the dates in readable format
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')

    # Filtering ratings and joining with to get genre and title 
    user_ratings_time = ratings[ratings['userId'] == user_id].merge(movies, on='movieId')

    # Getting the year from the rating timestamp column
    user_ratings_time['year'] = user_ratings_time['timestamp'].dt.year

    # Get the first genre
    user_ratings_time['genre'] = user_ratings_time['genres'].str.split('|').str[0]

    # Plot
    if st.checkbox("📈 See Genre Trends Over Time"):
        genre_trend = user_ratings_time.groupby(['year', 'genre']).size().reset_index(name='Count')

        # Drawing Line
        fig_trend = px.line(genre_trend, x='year', y='Count', color='genre',
                            title='Your Genre Interests Over Time')
        
        fig_trend.update_layout(
            yaxis=dict(visible=False), 
            xaxis_title='Year',
            title='Your Genre Interests Over Time',
            margin=dict(t=40, b=20)
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    #######################################
    # Fourth Section: Filtered Scatter Plot
    #######################################

    # ------------------ Scatter Plot: Predicting movie by predicted rating and genre------------------# 

    st.markdown("### 🎛️ For Our Most Selective Users: Pick Your Predicted Rating")

    # Setting Min and Max rating values
    min_rating = st.slider("Minimum Predicted Rating", min_value=0.0, max_value=5.0, step=0.1, value=3.0)

    # Dropdown multiselect to allow users pick their genres as well
    multi_genres = st.multiselect("Select Genres", options=available_genres)

    # Filtering based on what I just explained
    filtered_custom = filtered_movies[
        (filtered_movies['Predicted Rating'] >= min_rating) & 
        (filtered_movies['genre_list'].isin(multi_genres) if multi_genres else True)
    ]

    # Putting it all in correct order
    filtered_custom = filtered_custom.sort_values(by='Predicted Rating', ascending=False)
    filtered_custom['title'] = pd.Categorical(filtered_custom['title'], categories=filtered_custom['title'], ordered=True)

    # Checking if empty chart
    if filtered_custom.empty:
        st.write("No movies to display with current filters.")
    else:
        # Fixed Dot size
        dot_sizes = [30] * len(filtered_custom)

        fig = px.scatter(
            filtered_custom,
            x='Predicted Rating',
            y='title',
            title='👌 Your Unique Choice',
            # Big Dot Sizes
            size=dot_sizes,
            size_max=30,

            color_discrete_sequence=['steelblue']
        )

        fig.update_layout(
            xaxis_title="Predicted Rating",

            yaxis=dict(
                # Lets get rid off the Y axis title
                title=None,

                # Higher rating on top
                autorange='reversed',

                #Big Fotn size
                tickfont=dict(size=20) 
            ),
            
            xaxis=dict(
                # X-axis (Predicted Rating) font size
                tickfont=dict(size=16) 
            ),
        )

        st.plotly_chart(fig, use_container_width=True)

# ------------------------------- Analytics Page -------------------------------- #

# Sames as the Jupyter file ones
# See Jupyter file why the selection of the given Charts

elif page == "Analytics":
    st.title("📊 Movie Data Analytics Dashboard")

    # Setting Dark Mode Globally
    plt.style.use('dark_background')
    sns.set_theme(style="darkgrid", rc={
        'axes.facecolor': '#111111',
        'figure.facecolor': '#111111',
        'axes.labelcolor': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
        'text.color': 'white',
        'axes.edgecolor': 'white'
    })

    # 1. Distribution of Movie Ratings
    st.subheader("Distribution of Movie Ratings")
    fig, ax = plt.subplots()
    sns.histplot(merged['rating'], bins=10, kde=True, color='skyblue', edgecolor='white', ax=ax)
    ax.set_title("Distribution of Movie Ratings")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # 2. Top 10 Most Frequent Tags
    st.subheader("Top 10 Most Frequent Tags")
    top_tags = tags['tag'].value_counts().head(10)
    fig, ax = plt.subplots()
    sns.barplot(x=top_tags.index, y=top_tags.values, palette='crest', ax=ax)
    ax.set_title("Top 10 Most Frequent Tags")
    ax.set_xlabel("Tag")
    ax.set_ylabel("Frequency")
    ax.tick_params(axis='x', rotation=45, colors='white')
    st.pyplot(fig)

    # 3. Box Plot for Ratings
    st.subheader("Boxplot of Movie Ratings")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(y=merged['rating'], ax=ax, color='deepskyblue')
    ax.set_title("Boxplot of Movie Ratings", fontsize=14)
    ax.set_ylabel("Rating", fontsize=12)
    st.pyplot(fig)

    # 4. Pie Chart of Most Common Genres
    st.subheader("Most Common Genres")

    # Getting one out the rows
    genre_counts = movies['genres'].str.split('|').explode().value_counts()
    labels = genre_counts.index
    sizes = genre_counts.values

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140,
           textprops={'fontsize': 10, 'color': 'white'})
    ax.set_title("Most Common Movie Genres", fontsize=14, color='white')
    ax.axis('equal')
    st.pyplot(fig)

    # 5. Trend Line: Average Rating by Year
    st.subheader("Trend of Average Movie Ratings by Release Year")

    # Getting the Release year by getting the 4 digits of the row
    # Passing them to float
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype(float)

    # Merging, getting the rate means and grouping by year for plotting
    ratings_with_year = merged.merge(movies[['movieId', 'year']], on='movieId')
    yearly_avg = ratings_with_year.groupby('year')['rating'].mean().reset_index().dropna().sort_values('year')

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=yearly_avg, x='year', y='rating', marker='o', color='cyan', linewidth=2, ax=ax)
    ax.set_title("Trend of Average Movie Ratings by Release Year", fontsize=16)
    ax.set_xlabel("Release Year", fontsize=12)
    ax.set_ylabel("Average Rating", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
