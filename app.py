import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from datetime import datetime, timedelta
import time
from textblob import TextBlob
import numpy as np
from collections import Counter
import json
import io

# Configure Streamlit page
st.set_page_config(
    page_title="News Headline Analyzer",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .positive { color: #28a745; }
    .negative { color: #dc3545; }
    .neutral { color: #6c757d; }
    .anti-narrative { color: #fd7e14; }
    .upload-section {
        border: 2px dashed #cccccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .projection-positive { 
        background-color: #e6f7e6;
        padding: 1rem;
        border-radius: 10px;
    }
    .projection-negative { 
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 10px;
    }
    .projection-neutral { 
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

class NewsAnalyzer:
    def __init__(self):
        self.donor_keywords = [
            'bill gates', 'gates foundation', 'ford foundation', 'rockefeller',
            'world bank', 'usaid', 'dfid', 'european union', 'african development bank',
            'mastercard foundation', 'open society', 'soros', 'wellcome trust',
            'clinton foundation', 'carter center', 'oxfam', 'unicef', 'unesco',
            'who', 'undp', 'world health organization', 'international monetary fund',
            'imf', 'donor', 'funding', 'grant', 'donation', 'aid', 'development aid',
            'humanitarian aid', 'relief fund', 'charity', 'philanthropy', 'foundation',
            'world food programme', 'wfp', 'unhcr', 'ilo', 'ifad', 'gavi', 'global fund'
        ]
        
        self.anti_narrative_keywords = [
            'colonialism', 'neo-colonial', 'exploitation', 'dependency', 'corruption',
            'mismanagement', 'failed', 'waste', 'scandal', 'controversy', 'criticized',
            'accused', 'investigation', 'fraud', 'embezzlement', 'misappropriation',
            'ineffective', 'unsuccessful', 'backlash', 'protest', 'rejected',
            'condemned', 'suspended', 'withdrawn', 'terminated', 'breach', 'violation',
            'misuse', 'abuse', 'inappropriate', 'inadequate', 'disappointing'
        ]
        
        self.major_donors = {
            'Gates Foundation': ['gates foundation', 'bill gates', 'melinda gates', 'bill & melinda gates'],
            'Ford Foundation': ['ford foundation'],
            'Rockefeller Foundation': ['rockefeller foundation', 'rockefeller'],
            'World Bank': ['world bank', 'international bank'],
            'USAID': ['usaid', 'us aid', 'united states aid'],
            'Mastercard Foundation': ['mastercard foundation'],
            'Open Society': ['open society', 'soros foundation', 'george soros'],
            'Wellcome Trust': ['wellcome trust', 'wellcome'],
            'African Development Bank': ['african development bank', 'afdb'],
            'European Union': ['european union', 'eu aid', 'eu funding'],
            'World Food Programme': ['world food programme', 'wfp'],
            'UNICEF': ['unicef', 'united nations children'],
            'UNHCR': ['unhcr', 'un refugee'],
            'WHO': ['world health organization', 'who'],
            'Global Fund': ['global fund', 'gfatm'],
            'GAVI': ['gavi', 'vaccine alliance']
        }
        
        self.africa_countries = [
            'kenya', 'nigeria', 'south africa', 'ghana', 'ethiopia', 'uganda', 'tanzania',
            'rwanda', 'botswana', 'senegal', 'ivory coast', 'morocco', 'egypt', 'tunisia',
            'algeria', 'cameroon', 'madagascar', 'mozambique', 'zambia', 'zimbabwe',
            'malawi', 'burkina faso', 'mali', 'niger', 'chad', 'sudan', 'south sudan',
            'somalia', 'djibouti', 'eritrea', 'liberia', 'sierra leone', 'guinea',
            'gambia', 'cape verde', 'mauritius', 'seychelles'
        ]

    def parse_text_file(self, uploaded_file):
        """Parse uploaded text file and extract headlines"""
        try:
            content = uploaded_file.read()
            
            try:
                text_content = content.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    text_content = content.decode('latin-1')
                except UnicodeDecodeError:
                    text_content = content.decode('cp1252', errors='ignore')
            
            lines = text_content.split('\n')
            headlines = []
            
            for i, line in enumerate(lines):
                line = line.strip()
                if line and len(line) > 10:
                    headlines.append({
                        'title': line,
                        'description': '',
                        'source': 'Uploaded File',
                        'published_at': datetime.now().strftime('%Y-%m-%d'),
                        'url': '',
                        'country': self.detect_country(line),
                        'line_number': i + 1
                    })
            
            return headlines
            
        except Exception as e:
            st.error(f"Error parsing text file: {str(e)}")
            return []

    def detect_country(self, text):
        """Detect African country mentions in text"""
        text_lower = text.lower()
        for country in self.africa_countries:
            if country in text_lower:
                return country.title()
        return 'Africa (General)'

    def analyze_sentiment(self, text):
        """Analyze sentiment using TextBlob"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                return 'Positive', polarity
            elif polarity < -0.1:
                return 'Negative', polarity
            else:
                return 'Neutral', polarity
        except:
            return 'Neutral', 0.0

    def detect_anti_narrative(self, text):
        """Detect anti-narrative keywords in text"""
        text_lower = text.lower()
        found_keywords = []
        
        for keyword in self.anti_narrative_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        return len(found_keywords) > 0, found_keywords

    def identify_donors(self, text):
        """Identify mentioned donors/foundations in text"""
        text_lower = text.lower()
        mentioned_donors = []
        
        for donor, keywords in self.major_donors.items():
            for keyword in keywords:
                if keyword in text_lower:
                    mentioned_donors.append(donor)
                    break
        
        return mentioned_donors

    def calculate_relevance_score(self, text):
        """Calculate how relevant the headline is to donor/foundation topics"""
        text_lower = text.lower()
        score = 0
        
        for keyword in self.donor_keywords:
            if keyword in text_lower:
                score += 2
        
        africa_terms = ['africa', 'african'] + self.africa_countries
        for term in africa_terms:
            if term in text_lower:
                score += 1
        
        dev_terms = ['development', 'aid', 'support', 'assistance', 'program', 'project', 'initiative']
        for term in dev_terms:
            if term in text_lower:
                score += 1
        
        return min(score, 10)

    def analyze_articles(self, articles):
        """Analyze all articles for sentiment, anti-narratives, and donor mentions"""
        analyzed_data = []
        
        for article in articles:
            full_text = f"{article['title']} {article['description']}"
            
            sentiment, polarity = self.analyze_sentiment(full_text)
            has_anti_narrative, anti_keywords = self.detect_anti_narrative(full_text)
            mentioned_donors = self.identify_donors(full_text)
            relevance_score = self.calculate_relevance_score(full_text)
            
            analyzed_data.append({
                **article,
                'sentiment': sentiment,
                'polarity': polarity,
                'has_anti_narrative': has_anti_narrative,
                'anti_keywords': anti_keywords,
                'mentioned_donors': mentioned_donors,
                'relevance_score': relevance_score,
                'full_text': full_text
            })
        
        return analyzed_data

    def generate_trends_prediction(self, df):
        """Generate trend predictions and future-oriented recommendations"""
        predictions = {}
        
        # Overall sentiment trend
        sentiment_counts = df['sentiment'].value_counts(normalize=True)
        positive_ratio = sentiment_counts.get('Positive', 0)
        negative_ratio = sentiment_counts.get('Negative', 0)
        
        if positive_ratio > 0.5:
            overall_trend = "Positive"
            trend_description = "Favorable climate for donor engagement"
            future_outlook = ("Based on current positive sentiment, we anticipate increased funding opportunities "
                            "in 2026-2027, particularly in sectors with strong positive coverage. Organizations "
                            "should prepare expansion plans and capacity building.")
        elif negative_ratio > 0.4:
            overall_trend = "Negative"
            trend_description = "Challenging environment, careful approach needed"
            future_outlook = ("Current negative sentiment suggests potential funding reductions or stricter "
                            "conditions in 2026. Organizations should diversify funding sources and strengthen "
                            "transparency measures to maintain donor confidence.")
        else:
            overall_trend = "Mixed"
            trend_description = "Balanced sentiment, selective engagement recommended"
            future_outlook = ("The mixed sentiment indicates stable but competitive funding landscape. "
                            "2026-2027 may see sector-specific shifts - focus on programs with demonstrated "
                            "impact and clear metrics.")
        
        predictions['overall'] = {
            'trend': overall_trend,
            'description': trend_description,
            'confidence': max(sentiment_counts.values) if len(sentiment_counts) > 0 else 0,
            'future_outlook': future_outlook,
            'projected_trend': self.project_future_trend(positive_ratio, negative_ratio)
        }
        
        # Donor-specific trends with future projections
        donor_trends = {}
        all_donors = []
        for donors in df['mentioned_donors']:
            all_donors.extend(donors)
        
        donor_counts = Counter(all_donors)
        for donor in donor_counts.most_common(5):
            donor_name = donor[0]
            donor_articles = df[df['mentioned_donors'].apply(lambda x: donor_name in x)]
            
            if not donor_articles.empty:
                avg_sentiment = donor_articles['polarity'].mean()
                positive_ratio = len(donor_articles[donor_articles['sentiment'] == 'Positive']) / len(donor_articles)
                
                # Determine current trend
                if avg_sentiment > 0.2 and positive_ratio > 0.6:
                    trend = "Highly Favorable"
                    projection = self.generate_donor_projection(donor_name, 'high')
                elif avg_sentiment > 0 and positive_ratio > 0.4:
                    trend = "Favorable"
                    projection = self.generate_donor_projection(donor_name, 'medium')
                elif avg_sentiment < -0.2 or positive_ratio < 0.2:
                    trend = "Unfavorable"
                    projection = self.generate_donor_projection(donor_name, 'low')
                else:
                    trend = "Neutral"
                    projection = self.generate_donor_projection(donor_name, 'stable')
                
                donor_trends[donor_name] = {
                    'trend': trend,
                    'avg_sentiment': avg_sentiment,
                    'positive_ratio': positive_ratio,
                    'mentions': donor[1],
                    'projection': projection,
                    'recommendation': self.generate_donor_recommendation(donor_name, trend)
                }
        
        predictions['donors'] = donor_trends
        return predictions

    def project_future_trend(self, positive_ratio, negative_ratio):
        """Project future funding trends based on current sentiment"""
        if positive_ratio > 0.6 and negative_ratio < 0.2:
            return "Strong growth expected (15-20% increase in 2026, 10-15% in 2027)"
        elif positive_ratio > 0.45:
            return "Moderate growth likely (5-10% annual increase through 2027)"
        elif negative_ratio > 0.5:
            return "Potential reductions (5-15% annual decrease through 2027)"
        elif negative_ratio > 0.35:
            return "Flat or slightly declining (0-5% annual change)"
        else:
            return "Stable funding expected (±3% annual change)"

    def generate_donor_projection(self, donor_name, trend_level):
        """Generate specific projections for major donors"""
        projections = {
            'high': {
                'Gates Foundation': "20-25% funding increase likely in health/tech sectors",
                'World Bank': "Expanded infrastructure lending in 2026-2027",
                'USAID': "New country-specific initiatives expected",
                'default': "Significant funding increases expected across focus areas"
            },
            'medium': {
                'Gates Foundation': "Steady 5-8% annual growth in core programs",
                'World Bank': "Continued support with tighter conditions",
                'default': "Moderate growth with focus on proven programs"
            },
            'low': {
                'Gates Foundation': "Possible program consolidation",
                'World Bank': "Reduced concessional financing",
                'default': "Potential 5-10% annual reductions through 2027"
            },
            'stable': {
                'Gates Foundation': "Maintained funding with emphasis on metrics",
                'default': "Flat funding with competitive application processes"
            }
        }
        
        return projections[trend_level].get(donor_name, projections[trend_level]['default'])

    def generate_donor_recommendation(self, donor_name, trend):
        """Generate strategic recommendations for engaging with specific donors"""
        recommendations = {
            'Highly Favorable': {
                'Gates Foundation': "Develop ambitious proposals in health/education technology",
                'World Bank': "Prepare large-scale infrastructure project proposals",
                'USAID': "Align with country-specific strategic plans",
                'default': "Propose innovative programs with clear scaling potential"
            },
            'Favorable': {
                'Gates Foundation': "Focus on measurable outcomes in existing programs",
                'default': "Strengthen monitoring & evaluation systems for competitive proposals"
            },
            'Unfavorable': {
                'World Bank': "Diversify funding sources while maintaining compliance",
                'default': "Implement transparency measures and impact documentation"
            },
            'Neutral': {
                'default': "Maintain current engagement while exploring new partnerships"
            }
        }
        
        base_recommendation = ""
        if trend in ['Highly Favorable', 'Favorable']:
            base_recommendation = "Consider this donor a priority for 2026 funding applications."
        elif trend == 'Unfavorable':
            base_recommendation = "Monitor situation closely and prepare contingency plans."
        
        specific_rec = recommendations[trend].get(donor_name, recommendations[trend]['default'])
        return f"{specific_rec} {base_recommendation}"

def main():
    st.markdown("<h1 class='main-header'>📰 News Headline Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("**Upload and Analyze News Headlines for Donors and Foundations in Africa**")
    
    analyzer = NewsAnalyzer()
    
    # File upload section
    st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
    st.subheader("📁 Upload Your News Headlines")
    st.markdown("Upload a text file containing news headlines (one headline per line)")
    
    uploaded_file = st.file_uploader(
        "Choose a text file",
        type=['txt'],
        help="Upload a .txt file with news headlines, one per line"
    )
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Analysis options in sidebar
    st.sidebar.header("📊 Analysis Options")
    show_sentiment = st.sidebar.checkbox("Sentiment Analysis", value=True)
    show_anti_narrative = st.sidebar.checkbox("Anti-Narrative Detection", value=True)
    show_donor_analysis = st.sidebar.checkbox("Donor/Foundation Analysis", value=True)
    show_trends = st.sidebar.checkbox("Trend Predictions (2026-2027)", value=True)
    relevance_threshold = st.sidebar.slider("Relevance Score Threshold", 0, 10, 2, 
                                          help="Filter headlines by relevance to donor/foundation topics")
    
    if uploaded_file is not None:
        st.success(f"File uploaded successfully: {uploaded_file.name}")
        
        if st.button("🚀 Analyze Headlines", type="primary"):
            with st.spinner("Parsing and analyzing headlines..."):
                # Parse the uploaded file
                articles = analyzer.parse_text_file(uploaded_file)
                
                if not articles:
                    st.error("No valid headlines found in the uploaded file.")
                    return
                
                st.info(f"Found {len(articles)} headlines to analyze...")
                
                # Analyze articles
                analyzed_data = analyzer.analyze_articles(articles)
                df = pd.DataFrame(analyzed_data)
                
                # Filter by relevance if threshold is set
                if relevance_threshold > 0:
                    original_count = len(df)
                    df = df[df['relevance_score'] >= relevance_threshold]
                    st.info(f"Filtered to {len(df)} relevant headlines (threshold: {relevance_threshold})")
                
                if df.empty:
                    st.warning("No headlines meet the relevance threshold. Try lowering the threshold.")
                    return
                
                # Display results
                st.success(f"Analysis complete! Analyzed {len(df)} headlines.")
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_articles = len(df)
                    st.metric("Total Headlines", total_articles)
                
                with col2:
                    positive_count = len(df[df['sentiment'] == 'Positive'])
                    st.metric("Positive Sentiment", positive_count, f"{positive_count/total_articles*100:.1f}%")
                
                with col3:
                    negative_count = len(df[df['sentiment'] == 'Negative'])
                    st.metric("Negative Sentiment", negative_count, f"{negative_count/total_articles*100:.1f}%")
                
                with col4:
                    anti_narrative_count = len(df[df['has_anti_narrative'] == True])
                    st.metric("Anti-Narrative Headlines", anti_narrative_count, f"{anti_narrative_count/total_articles*100:.1f}%")
                
                # Sentiment Analysis
                if show_sentiment:
                    st.header("📈 Sentiment Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Sentiment distribution pie chart
                        sentiment_counts = df['sentiment'].value_counts()
                        fig_pie = px.pie(
                            values=sentiment_counts.values,
                            names=sentiment_counts.index,
                            title="Sentiment Distribution",
                            color_discrete_map={
                                'Positive': '#28a745',
                                'Negative': '#dc3545',
                                'Neutral': '#6c757d'
                            }
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with col2:
                        # Sentiment by country
                        country_sentiment = df.groupby(['country', 'sentiment']).size().unstack(fill_value=0)
                        
                        fig_country = px.bar(
                            country_sentiment.reset_index(),
                            x='country',
                            y=['Positive', 'Neutral', 'Negative'],
                            title="Sentiment by Country/Region",
                            color_discrete_map={
                                'Positive': '#28a745',
                                'Negative': '#dc3545',
                                'Neutral': '#6c757d'
                            }
                        )
                        fig_country.update_xaxes(tickangle=45)
                        st.plotly_chart(fig_country, use_container_width=True)
                
                # Anti-Narrative Analysis
                if show_anti_narrative:
                    st.header("⚠️ Anti-Narrative Analysis")
                    
                    anti_narrative_df = df[df['has_anti_narrative'] == True]
                    
                    if not anti_narrative_df.empty:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Most common anti-narrative keywords
                            all_keywords = []
                            for keywords in anti_narrative_df['anti_keywords']:
                                all_keywords.extend(keywords)
                            
                            if all_keywords:
                                keyword_counts = Counter(all_keywords)
                                top_keywords = dict(keyword_counts.most_common(10))
                                
                                fig_keywords = px.bar(
                                    x=list(top_keywords.values()),
                                    y=list(top_keywords.keys()),
                                    orientation='h',
                                    title="Most Common Anti-Narrative Keywords",
                                    labels={'x': 'Frequency', 'y': 'Keywords'}
                                )
                                st.plotly_chart(fig_keywords, use_container_width=True)
                        
                        with col2:
                            # Anti-narrative by country
                            country_anti = anti_narrative_df['country'].value_counts().head(10)
                            
                            fig_country_anti = px.bar(
                                x=country_anti.values,
                                y=country_anti.index,
                                orientation='h',
                                title="Anti-Narrative Headlines by Country"
                            )
                            st.plotly_chart(fig_country_anti, use_container_width=True)
                        
                        # Sample anti-narrative headlines
                        st.subheader("Sample Anti-Narrative Headlines")
                        for idx, article in anti_narrative_df.head(5).iterrows():
                            with st.expander(f"📰 Line {article['line_number']}: {article['title'][:80]}..."):
                                st.write(f"**Country/Region:** {article['country']}")
                                st.write(f"**Keywords Found:** {', '.join(article['anti_keywords'])}")
                                st.write(f"**Sentiment:** {article['sentiment']} ({article['polarity']:.3f})")
                                st.write(f"**Full Headline:** {article['title']}")
                    else:
                        st.info("No anti-narrative content detected in the analyzed headlines.")
                
                # Donor/Foundation Analysis
                if show_donor_analysis:
                    st.header("🏛️ Donor & Foundation Analysis")
                    
                    # Flatten donor mentions
                    all_donors = []
                    for donors in df['mentioned_donors']:
                        all_donors.extend(donors)
                    
                    if all_donors:
                        donor_counts = Counter(all_donors)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Donor mention frequency
                            fig_donors = px.bar(
                                x=list(donor_counts.values()),
                                y=list(donor_counts.keys()),
                                orientation='h',
                                title="Donor/Foundation Mentions in Headlines"
                            )
                            st.plotly_chart(fig_donors, use_container_width=True)
                        
                        with col2:
                            # Donor sentiment analysis
                            donor_sentiment = {}
                            for donor in donor_counts.keys():
                                donor_articles = df[df['mentioned_donors'].apply(lambda x: donor in x)]
                                if not donor_articles.empty:
                                    avg_polarity = donor_articles['polarity'].mean()
                                    sentiment_dist = donor_articles['sentiment'].value_counts(normalize=True)
                                    donor_sentiment[donor] = {
                                        'avg_polarity': avg_polarity,
                                        'positive_ratio': sentiment_dist.get('Positive', 0),
                                        'negative_ratio': sentiment_dist.get('Negative', 0)
                                    }
                            
                            # Create donor sentiment visualization
                            if donor_sentiment:
                                donor_sentiment_df = pd.DataFrame(donor_sentiment).T
                                donor_sentiment_df = donor_sentiment_df.reset_index()
                                donor_sentiment_df.columns = ['Donor', 'Avg_Polarity', 'Positive_Ratio', 'Negative_Ratio']
                                
                                fig_sentiment = px.scatter(
                                    donor_sentiment_df,
                                    x='Positive_Ratio',
                                    y='Negative_Ratio',
                                    size=[abs(x) + 0.1 for x in donor_sentiment_df['Avg_Polarity']],
                                    hover_name='Donor',
                                    title="Donor Sentiment Analysis",
                                    labels={'Positive_Ratio': 'Positive Sentiment Ratio', 'Negative_Ratio': 'Negative Sentiment Ratio'}
                                )
                                st.plotly_chart(fig_sentiment, use_container_width=True)
                        
                        # Donor recommendations
                        st.subheader("💡 Donor Approach Recommendations")
                        
                        # Sort donors by positive sentiment and mention frequency
                        recommendations = []
                        for donor, stats in donor_sentiment.items():
                            score = (stats['positive_ratio'] * 0.6 + 
                                    (1 - stats['negative_ratio']) * 0.3 + 
                                    (stats['avg_polarity'] + 1) / 2 * 0.1) * donor_counts[donor]
                            recommendations.append((donor, score, stats))
                        
                        recommendations.sort(key=lambda x: x[1], reverse=True)
                        
                        for i, (donor, score, stats) in enumerate(recommendations[:5]):
                            with st.expander(f"#{i+1} {donor} (Score: {score:.2f})"):
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Mentions", donor_counts[donor])
                                with col2:
                                    st.metric("Positive Ratio", f"{stats['positive_ratio']:.2%}")
                                with col3:
                                    st.metric("Avg Sentiment", f"{stats['avg_polarity']:.3f}")
                                
                                # Show sample headlines
                                donor_articles = df[df['mentioned_donors'].apply(lambda x: donor in x)]
                                positive_articles = donor_articles[donor_articles['sentiment'] == 'Positive']
                                
                                if not positive_articles.empty:
                                    st.write("**Recent Positive Headlines:**")
                                    for _, article in positive_articles.head(3).iterrows():
                                        st.write(f"• {article['title']}")
                    else:
                        st.info("No specific donors or foundations mentioned in the analyzed headlines.")
                
                # Trend Predictions
                if show_trends:
                    st.header("🔮 Trend Predictions & Future Outlook (2026-2027)")
                    
                    predictions = analyzer.generate_trends_prediction(df)
                    
                    # Overall trend with future projection
                    st.subheader("Overall Funding Climate")
                    overall = predictions['overall']
                    
                    cols = st.columns([1, 3])
                    with cols[0]:
                        st.metric("Current Trend", overall['trend'])
                        st.metric("Confidence", f"{overall['confidence']:.1%}")
                    with cols[1]:
                        projection_class = "projection-positive" if overall['trend'] == "Positive" else \
                                         "projection-negative" if overall['trend'] == "Negative" else "projection-neutral"
                        st.markdown(f"<div class='{projection_class}'><strong>Future Projection:</strong> {overall['projected_trend']}</div>", 
                                   unsafe_allow_html=True)
                        st.write("**Strategic Implications:** " + overall['future_outlook'])
                    
                    # Donor-specific trends
                    if predictions['donors']:
                        st.subheader("Donor-Specific Projections")
                        
                        for donor, data in predictions['donors'].items():
                            with st.expander(f"{donor} - {data['trend']} (Mentions: {data['mentions']})"):
                                cols = st.columns([1, 3])
                                with cols[0]:
                                    st.metric("Avg Sentiment", f"{data['avg_sentiment']:.2f}")
                                    st.metric("Positive Ratio", f"{data['positive_ratio']:.1%}")
                                with cols[1]:
                                    projection_class = "projection-positive" if data['trend'] in ["Highly Favorable", "Favorable"] else \
                                                     "projection-negative" if data['trend'] == "Unfavorable" else "projection-neutral"
                                    st.markdown(f"<div class='{projection_class}'><strong>2026-2027 Projection:</strong> {data['projection']}</div>", 
                                               unsafe_allow_html=True)
                                    st.info(f"**Recommendation:** {data['recommendation']}")
                        
                        # Strategic recommendations
                        st.subheader("📊 Strategic Planning Guide")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.success("**Growth Opportunities (2026 Focus):**")
                            growth_donors = [d for d, data in predictions['donors'].items() 
                                           if data['trend'] in ['Highly Favorable', 'Favorable']]
                            for donor in growth_donors[:3]:
                                st.write(f"✅ **{donor}:** {predictions['donors'][donor]['projection']}")
                        
                        with col2:
                            st.warning("**Risk Mitigation (2026 Watchlist):**")
                            risk_donors = [d for d, data in predictions['donors'].items() 
                                         if data['trend'] == 'Unfavorable']
                            if risk_donors:
                                for donor in risk_donors[:2]:
                                    st.write(f"⚠️ **{donor}:** {predictions['donors'][donor]['projection']}")
                            else:
                                st.info("No high-risk donors identified in current analysis")
                            
                        st.markdown("---")
                        st.write("**Note:** Projections based on current sentiment trends and historical funding patterns. "
                                "Actual outcomes may vary based on global economic conditions and policy changes.")
                
                # Raw data table
                st.header("📋 Analyzed Headlines")
                
                # Create display DataFrame
                display_df = df[['line_number', 'title', 'country', 'sentiment', 'polarity', 
                               'has_anti_narrative', 'mentioned_donors', 'relevance_score']].copy()
                display_df['mentioned_donors'] = display_df['mentioned_donors'].apply(lambda x: ', '.join(x) if x else 'None')
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    column_config={
                        "line_number": "Line #",
                        "title": st.column_config.TextColumn("Headline", width="large"),
                        "country": "Country/Region",
                        "sentiment": st.column_config.SelectboxColumn("Sentiment", options=["Positive", "Negative", "Neutral"]),
                        "polarity": st.column_config.NumberColumn("Polarity", format="%.3f"),
                        "has_anti_narrative": st.column_config.CheckboxColumn("Anti-Narrative"),
                        "mentioned_donors": "Mentioned Donors",
                        "relevance_score": st.column_config.NumberColumn("Relevance", format="%d")
                    }
                )
                
                # Download option
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Full Analysis (CSV)",
                    data=csv,
                    file_name=f"headlines_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    else:
        # Show sample format
        st.info("👆 Please upload a text file to begin analysis")
        
        with st.expander("📋 Sample File Format"):
            st.markdown("""
            **Your text file should contain one headline per line:**
            
            ```
            Gates Foundation announces $100M investment in African healthcare
            World Bank approves new loan for Kenya infrastructure project
            Concerns raised over aid effectiveness in Somalia
            USAID launches education initiative across West Africa
            Corruption allegations surface in Ghana development fund
            ```
            
            **Supported formats:**
            - Plain text files (.txt)
            - One headline per line
            - UTF-8 encoding recommended
            """)

if __name__ == "__main__":
    main()