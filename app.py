import streamlit as st
import plotly.express as px
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import altair as alt
from Backend.reddit_sentiment_analysis import get_posts_and_comments, analyze_sentiment

# Set Streamlit page configuration
st.set_page_config(page_title="Reddit Sentiment Analysis Dashboard", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .reportview-container .markdown-text-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .dataframe th, .dataframe td {
        padding: 10px;
        text-align: left;
        border: 1px solid #ddd;
    }
    .dataframe th {
        background-color: #f4f4f4;
        font-weight: bold;
    }
    /* Footer styling */
    .footer {
        position: relative;
        bottom: 0;
        width: 100%;
        background-color: #ffffff;
        color: white;
        text-align: center;
        padding: 10px 0;
        margin-top: 50px;
    }
    .footer a {
        color: white;
        text-decoration: none;
        margin: 0 15px;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    .footer img {
        width: 30px;
        height: 30px;
        vertical-align: middle;
    }
    </style>
    """, unsafe_allow_html=True)

def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

def main():
    st.title("Reddit Sentiment Analysis Dashboard")
    st.sidebar.header("Configuration")

    subreddit_name = st.sidebar.text_input("Enter Subreddit Name:", "elonmusk")
    post_limit = st.sidebar.slider("Number of Posts to Analyze:", 1, 100, 10)
    comment_limit = st.sidebar.slider("Number of Comments per Post:", 1, 100, 10)

    if st.sidebar.button("Run Analysis"):
        with st.spinner('Fetching data and performing analysis...'):
            try:
                posts_data = get_posts_and_comments(subreddit_name, post_limit, comment_limit)
                if posts_data:
                    st.success(f"Fetched {len(posts_data)} posts from r/{subreddit_name}")

                    # DataFrame to hold sentiments
                    sentiments = []

                    for idx, post in enumerate(posts_data, start=1):
                        post_sentiment, post_score = analyze_sentiment(post['title'])
                        sentiments.append({
                            'Post': post['title'],
                            'Comment': None,
                            'Sentiment': post_sentiment,
                            'Score': post_score
                        })

                        for comment in post['comments']:
                            comment_sentiment, comment_score = analyze_sentiment(comment)
                            sentiments.append({
                                'Post': None,
                                'Comment': comment,
                                'Sentiment': comment_sentiment,
                                'Score': comment_score
                            })

                    # Convert to DataFrame
                    df = pd.DataFrame(sentiments)

                    # Sentiment Distribution Pie Chart
                    sentiment_counts = df['Sentiment'].value_counts().reset_index()
                    sentiment_counts.columns = ['Sentiment', 'Count']
                    fig_pie = px.pie(sentiment_counts, names='Sentiment', values='Count', title='Sentiment Distribution')
                    st.plotly_chart(fig_pie, use_container_width=True)

                    # Word Cloud for Posts and Comments
                    all_text = ' '.join(df['Post'].dropna().tolist() + df['Comment'].dropna().tolist())
                    fig_wordcloud = generate_wordcloud(all_text)
                    st.pyplot(fig_wordcloud)

                    # Display DataFrame with Conditional Formatting
                    def highlight_sentiment(row):
                        color = '#ffcccc' if row.Sentiment == 'Negative' else '#ccffcc' if row.Sentiment == 'Positive' else '#ffffcc'
                        return ['background-color: {}'.format(color)] * len(row)

                    st.dataframe(df.style.apply(highlight_sentiment, axis=1))

                    # Bar Chart for Sentiment Scores
                    # Limit content length for better visualization
                    df['Content'] = df.apply(lambda row: row['Post'] if pd.notna(row['Post']) else row['Comment'], axis=1)
                    df['Short_Content'] = df['Content'].apply(lambda x: x[:50] + '...' if len(x) > 50 else x)

                    bar_chart = alt.Chart(df).mark_bar().encode(
                        x=alt.X('Short_Content:N', title='Content', sort=None),
                        y=alt.Y('Score:Q', title='Sentiment Score'),
                        color=alt.Color('Sentiment:N', legend=None),
                        tooltip=[alt.Tooltip('Content:N', title='Full Content'), alt.Tooltip('Score:Q', title='Sentiment Score')]
                    ).properties(
                        width=800,
                        height=400,
                        title='Sentiment Scores for Posts and Comments'
                    ).configure_axis(
                        labelAngle=-45
                    )

                    st.altair_chart(bar_chart, use_container_width=True)

                else:
                    st.warning(f"No posts found for subreddit: r/{subreddit_name}")
            except Exception as e:
                st.error(f"An error occurred: {e}")

    # Footer with contact information
    footer = """
    <div class="footer">
        <p>Connect with me:</p>
        <a href="https://github.com/YourGitHubUsername" target="_blank">
            <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Octicons-mark-github.svg" alt="GitHub">
        </a>
        <a href="https://www.linkedin.com/in/YourLinkedInUsername" target="_blank">
            <img src="https://www.svgrepo.com/show/448234/linkedin.svg" alt="LinkedIn">
        </a>
    </div>
    """
    st.markdown(footer, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
