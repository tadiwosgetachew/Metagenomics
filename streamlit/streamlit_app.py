import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import colorsys
import os
import sys
import os
import streamlit as st

st.write("Current working directory:", os.getcwd())

st.set_page_config(layout="wide")
st.title("Crohn's Microbiome Analysis Dashboard")
st.markdown(""" 
This dashboard explores microbiome differences between Crohn's disease and control groups. It begins with Alpha diversity (Faith’s PD, Shannon Index) to assess within-sample diversity. Next, Beta diversity is shown using a 3D Unweighted UniFrac PCoA, highlighting between-group separation. The ANCOM volcano plot then identifies taxa with significant differential abundance. Lastly, a stacked bar plot visualizes the relative abundance of taxa per sample, grouped by disease status.
""")


# ---------- Utilities ----------
def generate_shades(hex_color, n_shades=2):
    rgb = tuple(int(hex_color.lstrip('#')[i:i+2], 16)/255.0 for i in (0,2,4))
    hls = colorsys.rgb_to_hls(*rgb)
    variations = []
    for delta in [-0.15, 0.15][:n_shades]:
        new_l = max(0, min(1, hls[1] + delta))
        new_rgb = colorsys.hls_to_rgb(hls[0], new_l, hls[2])
        variations.append('#{:02X}{:02X}{:02X}'.format(int(new_rgb[0]*255), int(new_rgb[1]*255), int(new_rgb[2]*255)))
    return variations

def safe_print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', errors='replace').decode())

# ---------- Faith PD ----------
st.header("1. Alpha Diversity")
st.subheader("1.1. Faith's Phylogenetic Diversity by Disease Group")
try:
    dfpd = pd.read_csv("data/faith_pd_group_significance.tsv", sep="\t", comment='#')    
    fig = px.box(
        dfpd, 
        x='disease_group', 
        y='faith_pd', 
        color='disease_group',
        color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c'],  # contrasting color palette
        #title="Faith's PD by Disease Group", 
        height=600,
        points='all'
    )

    fig.update_layout(showlegend=False)
    fig.update_xaxes(showgrid=False, zeroline=False, showline=True, linewidth=1.5,
                     linecolor='black', ticks='outside', tickfont=dict(size=14),
                     title=dict(text='Disease Group', font=dict(size=16)))
    fig.update_yaxes(showgrid=False, zeroline=False, showline=True, linewidth=1.5,
                     linecolor='black', ticks='outside', tickfont=dict(size=14),
                     title=dict(text="Faith's PD", font=dict(size=16)))

    st.plotly_chart(fig, use_container_width=True)

    kruskal_html = """
    <h3>Kruskal-Wallis (All Groups)</h3>
    <table style="border-collapse: collapse; border: 1px solid #ccc; font-size: 16px;">
    <tr style="background-color: #f2f2f2;">
    <th style="padding: 8px; border: 1px solid #ccc;">H</th>
    <th style="padding: 8px; border: 1px solid #ccc;">p-value</th>
    </tr>
    <tr>
    <td style="padding: 8px; border: 1px solid #ccc;">9.2956</td>
    <td style="padding: 8px; border: 1px solid #ccc;">0.0023</td>
    </tr>
    </table>

    <h3>Kruskal-Wallis (Pairwise)</h3>
    <table style="border-collapse: collapse; border: 1px solid #ccc; font-size: 16px;">
    <tr style="background-color: #f2f2f2;">
    <th style="padding: 8px; border: 1px solid #ccc;">Group 1</th>
    <th style="padding: 8px; border: 1px solid #ccc;">Group 2</th>
    <th style="padding: 8px; border: 1px solid #ccc;">H</th>
    <th style="padding: 8px; border: 1px solid #ccc;">p-value</th>
    <th style="padding: 8px; border: 1px solid #ccc;">q-value</th>
    </tr>
    <tr>
    <td style="padding: 8px; border: 1px solid #ccc;">CD (n=22)</td>
    <td style="padding: 8px; border: 1px solid #ccc;">Control (n=28)</td>
    <td style="padding: 8px; border: 1px solid #ccc;">9.2956</td>
    <td style="padding: 8px; border: 1px solid #ccc;">0.0023</td>
    <td style="padding: 8px; border: 1px solid #ccc;">0.0023</td>
    </tr>
    </table>
    """
    st.markdown(kruskal_html, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Error Faith PD: {e}")


st.markdown("<div style='margin: 40px 0;'></div>", unsafe_allow_html=True)   #Add space between this and the next section


# ---------- Shannon Diversity ----------
st.subheader("1.2. Shannon Diversity Index by Disease Group")
try:
    dfsh = pd.read_csv("data/shannon_group_significance.tsv", sep="\t", comment='#')
    
    fig = px.box(
        dfsh, 
        x='disease_group', 
        y='shannon_entropy', 
        color='disease_group',
        color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c'],  # contrasting color palette
        #title="Shannon Diversity by Disease Group", 
        height=600,
        points='all'
    )

    fig.update_layout(showlegend=False)
    fig.update_xaxes(showgrid=False, zeroline=False, showline=True, linewidth=1.5,
                     linecolor='black', ticks='outside', tickfont=dict(size=14),
                     title=dict(text='Disease Group', font=dict(size=16)))
    fig.update_yaxes(showgrid=False, zeroline=False, showline=True, linewidth=1.5,
                     linecolor='black', ticks='outside', tickfont=dict(size=14),
                     title=dict(text="Shannon Diversity", font=dict(size=16)))

    st.plotly_chart(fig, use_container_width=True)

    # Kruskal-Wallis stats (Shannon Diversity)
    shannon_html = """
    <h3>Kruskal-Wallis (All Groups)</h3>
    <table style="border-collapse: collapse; border: 1px solid #ccc; font-size: 16px;">
    <tr style="background-color: #f2f2f2;">
    <th style="padding: 8px; border: 1px solid #ccc;">H</th>
    <th style="padding: 8px; border: 1px solid #ccc;">p-value</th>
    </tr>
    <tr>
    <td style="padding: 8px; border: 1px solid #ccc;">9.176853</td>
    <td style="padding: 8px; border: 1px solid #ccc;">0.002451</td>
    </tr>
    </table>
    
    <h3>Kruskal-Wallis (Pairwise)</h3>
    <table style="border-collapse: collapse; border: 1px solid #ccc; font-size: 16px;">
    <tr style="background-color: #f2f2f2;">
    <th style="padding: 8px; border: 1px solid #ccc;">Group 1</th>
    <th style="padding: 8px; border: 1px solid #ccc;">Group 2</th>
    <th style="padding: 8px; border: 1px solid #ccc;">H</th>
    <th style="padding: 8px; border: 1px solid #ccc;">p-value</th>
    <th style="padding: 8px; border: 1px solid #ccc;">q-value</th>
    </tr>
    <tr>
    <td style="padding: 8px; border: 1px solid #ccc;">CD (n=22)</td>
    <td style="padding: 8px; border: 1px solid #ccc;">Control (n=28)</td>
    <td style="padding: 8px; border: 1px solid #ccc;">9.176853</td>
    <td style="padding: 8px; border: 1px solid #ccc;">0.002451</td>
    <td style="padding: 8px; border: 1px solid #ccc;">0.002451</td>
    </tr>
    </table>
    """
    st.markdown(shannon_html, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Error Shannon Diversity: {e}")

st.markdown("<div style='margin: 60px 0;'></div>", unsafe_allow_html=True)   #Add space between this and the next section

# ---------- 3D UniFrac PCoA ----------
st.header("2. Beta Diversity: Unweighted UniFrac PCoA")
try:
    def load_ordination(fp):
        with open(fp, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        variances = [None,None,None]
        for i, line in enumerate(lines):
            if line.strip().lower().startswith("proportion explained"):
                parts = lines[i+1].strip().split()
                if len(parts)>=3:
                    variances = [round(float(parts[j])*100,2) for j in range(3)]
                break
        start_idx = next((i for i,line in enumerate(lines) if line.strip().startswith("Site")), None)
        if start_idx is None:
            raise ValueError("No 'Site' section in ordination file")
        data=[]
        for line in lines[start_idx+1:]:
            parts=line.strip().split()
            if len(parts)>=4:
                try:
                    pc1 = float(parts[1].replace('\u2212','-'))
                    pc2 = float(parts[2].replace('\u2212','-'))
                    pc3 = float(parts[3].replace('\u2212','-'))
                    data.append([parts[0], pc1, pc2, pc3])
                except:
                    continue
        return pd.DataFrame(data, columns=["SampleID","PC1","PC2","PC3"]), variances

    def load_metadata(fp):
        dfm = pd.read_csv(fp, sep="\t", dtype=str)
        sample_col = next((c for c in dfm.columns if c.lower().replace('-','')=="sampleid"), None)
        if not sample_col:
            raise ValueError("No SampleID column in metadata")
        dfm.rename(columns={sample_col:"SampleID"}, inplace=True)
        return dfm

    def get_permanova_html():
        return """
        <hr><h3>Pairwise PERMANOVA Results</h3>
        <table border="1" cellpadding="5"><tr><th>Sample size</th><th>Permutations</th><th>Pseudo-F</th><th>p-value</th><th>q-value</th></tr>
        <tr><td>50</td><td>999</td><td>5.050342</td><td>0.001</td><td>0.001</td></tr></table>
        """

    fp_ord = "data/ordination.txt"
    fp_meta = "data/metadata.tsv"
    df_ord, variances = load_ordination(fp_ord)
    df_meta = load_metadata(fp_meta)
    dfm = pd.merge(df_ord, df_meta, on="SampleID", how="inner")
    if dfm.empty:
        raise ValueError("No overlapping SampleIDs")
    group_col = [c for c in dfm.columns if c not in ["SampleID","PC1","PC2","PC3"]][0]
    df_cd = dfm[dfm[group_col]=="CD"]
    df_ctrl = dfm[dfm[group_col]=="Control"]
    axis_min = min(dfm[["PC1","PC2","PC3"]].min().min(), 0)
    axis_max = dfm[["PC1","PC2","PC3"]].max().max()

    axis_style = dict(
    showticklabels=False,
    #tickfont=dict(size=14, color='black'),
    #titlefont=dict(size=16, color='black'),
    showgrid=False,
    zeroline=False,
    showline=True,
    linecolor="black",
    linewidth=4,
    #ticks='outside',
    #tickwidth=2,
    #tickcolor='black',
    showbackground=False,
    range=[axis_min, axis_max]
)

    trace1 = go.Scatter3d(x=df_cd["PC1"], y=df_cd["PC2"], z=df_cd["PC3"], mode='markers', name='CD',
                          marker=dict(size=6, color='red', opacity=0.9, line=dict(width=0.5, color='black')),
                          text=df_cd.apply(lambda r: f"SampleID: {r.SampleID}<br>Group: CD<br>PC1: {r.PC1:.3f}<br>PC2: {r.PC2:.3f}<br>PC3: {r.PC3:.3f}", axis=1),
                          hoverinfo="text")
    trace2 = go.Scatter3d(x=df_ctrl["PC1"], y=df_ctrl["PC2"], z=df_ctrl["PC3"], mode='markers', name='Control',
                          marker=dict(size=6, color='green', opacity=0.9, line=dict(width=0.5, color='black')),
                          text=df_ctrl.apply(lambda r: f"SampleID: {r.SampleID}<br>Group: Control<br>PC1: {r.PC1:.3f}<br>PC2: {r.PC2:.3f}<br>PC3: {r.PC3:.3f}", axis=1),
                          hoverinfo="text")
    fig = go.Figure(data=[trace1, trace2])
    fig.update_layout(
    scene=dict(
        xaxis=dict(title=f"PC1 ({variances[0]}%)", **axis_style),
        yaxis=dict(title=f"PC2 ({variances[1]}%)", **axis_style),
        zaxis=dict(title=f"PC3 ({variances[2]}%)", **axis_style),
        bgcolor="white"
    ),
    legend=dict(x=0.85, y=0.95, borderwidth=0, bgcolor='rgba(0,0,0,0)',  font=dict(size=16, color='black')),
    margin=dict(l=50, r=50, b=50, t=80),  # <<<<< Increase margins
    height=800,  # <<<<< Increase height
)

    st.plotly_chart(fig, use_container_width=False)
    st.markdown(get_permanova_html(), unsafe_allow_html=True)

except Exception as e:
    st.error(f"Error Beta‑diversity PCoA: {e}")

# Add space between this and the next section
st.markdown("<div style='margin: 60px 0;'></div>", unsafe_allow_html=True)

# ---------- ANCOM Volcano & Tables ----------
st.header("3. Differential Abundance")
try:
    dfv = pd.read_csv("data/ancom_volcano.tsv", sep="\t", header=None)    
    dfv.columns = ['Taxon','CLR F-statistic','W']
    dfv['W'] = pd.to_numeric(dfv['W'], errors='coerce')
    dfv['CLR F-statistic'] = pd.to_numeric(dfv['CLR F-statistic'], errors='coerce')
    dfv.dropna(subset=['W','CLR F-statistic'], inplace=True)
    threshold = 0.7 * dfv['W'].max()
    dfv['Reject null hypothesis'] = dfv['W'] >= threshold
    df_true = dfv[dfv['Reject null hypothesis']]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_true['CLR F-statistic'], y=df_true['W'], mode='markers', name='True',
                             marker=dict(color='red', size=9), text=df_true['Taxon'],
                             hovertemplate="<b>Taxon:</b> %{text}<br><b>CLR F-statistic:</b> %{x:.3f}<br><b>W:</b> %{y}<extra></extra>"))
    df_false = dfv[~dfv['Reject null hypothesis']]
    fig.add_trace(go.Scatter(x=df_false['CLR F-statistic'], y=df_false['W'], mode='markers', name='False',
                             marker=dict(color='#d3d3d3', size=9, line=dict(color='blue', width=1.5)),
                             text=df_false['Taxon'],
                             hovertemplate="<b>Taxon:</b> %{text}<br><b>CLR F-statistic:</b> %{x:.3f}<br><b>W:</b> %{y}<extra></extra>"))
    fig.add_hline(y=threshold, line_dash="dash", line_color="black")
    fig.update_layout(
    title="ANCOM Volcano Plot",
    height=600,
    width=1500,
    xaxis=dict(
        title=dict(text="CLR F-statistic", font=dict(size=16, color='black')),
        showgrid=False,
        zeroline=False,
        showline=True,
        linewidth=2,
        linecolor='black',
        ticks='outside',
        tickfont=dict(size=14)
    ),
    yaxis=dict(
        title=dict(text="W-statistic", font=dict(size=16, color='black')),
        showgrid=False,
        zeroline=False,
        showline=True,
        linewidth=2,
        linecolor='black',
        ticks='outside',
        tickfont=dict(size=14)
    ),
    plot_bgcolor='white',
    paper_bgcolor='white',
    hovermode='closest',
    legend=dict(
        title=dict(text="Reject null hypothesis", font=dict(size=16)),
        font=dict(size=16)
    )
)


    st.plotly_chart(fig, use_container_width=True)

    st.subheader("All Rejected Features")

    w_table = """
    
    <div style="overflow-x:auto;">
    <table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse; font-family: sans-serif; font-size: 16px; min-width: 600px;">
    <tr style="background-color: #f2f2f2;">
        <th>Taxon</th>
        <th>W</th>
        <th>CLR</th>
    </tr>
    """

    for _, row in df_true.sort_values(by='W', ascending=False).iterrows():
        w_table += f'<tr><td>{row["Taxon"]}</td><td>{int(row["W"])}</td><td>{row["CLR F-statistic"]:.3f}</td></tr>'

    w_table += '</table></div>'

    st.markdown(w_table, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Error ANCOM section: {e}")

try:
    # =========================
    # Load actual percentile abundance data
    # =========================
    percentile_file = "data/percent-abundances.tsv"
    df_percentile_raw = pd.read_csv(percentile_file, sep="\t", skiprows=[1])  # Skip second header row

    # Flatten headers manually (CD and Control groups with 0-100 percentiles)
    original_columns = df_percentile_raw.columns.tolist()
    fixed_columns = ['Taxon']
    for i, col in enumerate(original_columns[1:]):
        group = 'CD' if i < 5 else 'Control'
        percentile = ['0', '25', '50', '75', '100'][i % 5]
        fixed_columns.append(f"{group}_{percentile}")
    df_percentile_raw.columns = fixed_columns

    # Filter only rejected taxa
    df_pct = df_percentile_raw[df_percentile_raw['Taxon'].isin(df_true['Taxon'])].copy()

    # =========================
    # Build percentile table
    # =========================
    pct_table = '<h3>Percentile Abundances of Features by Group (All Rejected)</h3><table border="1" cellpadding="5" cellspacing="0">'
    pct_table += (
        '<tr><th rowspan="2">Taxon</th><th colspan="5">CD</th><th colspan="5">Control</th></tr>'
        '<tr><th>0%</th><th>25%</th><th>50%</th><th>75%</th><th>100%</th>'
        '<th>0%</th><th>25%</th><th>50%</th><th>75%</th><th>100%</th></tr>'
    )
    for _, row in df_pct.set_index('Taxon').loc[df_true.sort_values(by='W', ascending=False)['Taxon']].reset_index().iterrows():
        pct_table += f'<tr><td>{row["Taxon"]}</td>'
        for col in [
            "CD_0", "CD_25", "CD_50", "CD_75", "CD_100",
            "Control_0", "Control_25", "Control_50", "Control_75", "Control_100"
        ]:
            pct_table += f'<td>{row[col]}</td>'
        pct_table += '</tr>'
    pct_table += '</table>'

    st.markdown(f'<div style="overflow-x:auto;">{pct_table}</div>', unsafe_allow_html=True)

except Exception as e:
    st.error(f"Error ANCOM section: {e}")

st.markdown("<div style='margin: 60px 0;'></div>", unsafe_allow_html=True)   #Add space between this and the next section


# ---------- Bar Plot: Taxa ----------
st.header("4. Taxonomic Compositions")
try:
    df = pd.read_csv("data/taxa_bar_level_6.csv", index_col=0)
    df.reset_index(inplace=True)
    if 'index' in df.columns:
        df.rename(columns={'index':'SampleID'}, inplace=True)
    metadata_cols = ["host_age", "host_sex", "family_id", "Genetic_Risk_Score", "Dysbiosis_Score", "ethnicity"]
    df.drop(columns=[col for col in metadata_cols if col in df.columns], inplace=True, errors='ignore')
    if 'disease_group' not in df.columns:
        st.error("Missing 'disease_group' column in your dataset.")
    else:
        id_vars = ['SampleID','disease_group']
        taxa_cols = df.columns.difference(id_vars)
        df[taxa_cols] = df[taxa_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        df[taxa_cols] = df[taxa_cols].div(df[taxa_cols].sum(axis=1).replace(0,1), axis=0)*100
        df['TotalAbundance'] = df[taxa_cols].sum(axis=1)
        df = df.sort_values('TotalAbundance', ascending=True).drop(columns='TotalAbundance')
        long_df = df.melt(id_vars=id_vars, var_name='Taxa', value_name='RelativeAbundance')
        long_df['SampleLabel'] = long_df['SampleID'] + " (" + long_df['disease_group'] + ")"
        long_df['SampleLabel'] = pd.Categorical(long_df['SampleLabel'],
                                                categories=long_df['SampleLabel'].unique(),
                                                ordered=True)
        base_palette = ['#7FC97F','#BEAED4','#FDC086','#FFFF99','#386CB0','#F0027F','#BF5B17','#666666']
        extended_palette = []
        for color in base_palette:
            extended_palette.append(color)
            extended_palette.extend(generate_shades(color, n_shades=2))
        fig = px.bar(long_df, x='SampleLabel', y='RelativeAbundance', color='Taxa',
                     title='Stacked Taxa Bar Plot (Sorted by Total Abundance)',
                     template='plotly_white',
                     color_discrete_sequence=extended_palette,
                     height=1000, width=1800)
        fig.update_layout(barmode='stack', xaxis_tickangle=45,
                          xaxis_title='Sample (Disease Group)',
                          yaxis_title='Relative Abundance (%)',
                          yaxis=dict(ticksuffix='%', range=[0,100], tickformat='.0f'),
                          margin=dict(l=40, r=250, t=60, b=150),
                          legend=dict(title='Taxa', orientation='v', x=1.02, y=1,
                                      xanchor='left', yanchor='top'))
        

        # --- Render large Plotly figure with horizontal scroll ---
        plot_html = fig.to_html(include_plotlyjs='cdn', full_html=False)

        components.html(
        f"""
        <div style="width: 100%; overflow-x: auto;">
            <div style="min-width: 1900px;">{plot_html}</div>
        </div>
    """,
    height=1000,
    scrolling=True
)

except Exception as e:
    st.error(f"Error Taxa plot: {e}")






