"""
GEO Evaluation - Final 3 Plots + Save results to Excel
"""

import pandas as pd
import numpy as np
import json
import re
import os
import jieba
import jieba.posseg as pseg
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm
from math import pi

# ================== 1. Load product descriptions and extract keywords ==================
with open('Product Description.json', 'r', encoding='utf-8') as f:
    product_db = json.load(f)

STOPWORDS = set('的 了 和 与 或 是 在 有 就 都 而 及 等 对 为 以 于 由 这 那 它 她 他 我们 你们 它们 这个 那个 这些 那些 也是 还有 可以 进行 提供 使用 采用 拥有 具备 具有 获得 来自 经过 通过 作为 一种 一个 一样 一些 这款 该款 其 之 被 把 给 让 使 从 到 上 下 中 内 外 前 后 左右 上下 之间 分别 同时 并且 不仅 而且 或者 因此 所以 如果 那么 虽然 但是 因为 所以 由于 什么 怎么 怎样 为什么'.split())

def extract_short_keywords(text, top_n=12):
    if not text:
        return []
    words = []
    for word, flag in pseg.cut(text):
        word = word.strip()
        if len(word) < 2 or len(word) > 8:
            continue
        if word in STOPWORDS:
            continue
        if flag[0] in ('n', 'v', 'a', 'eng', 'm', 'x', 'i'):
            if word.isdigit():
                continue
            words.append(word)
    num_units = re.findall(r'(\d+(?:\.\d+)?)\s*(?:元|块|毫升|ml|毫米|mm|克|g|千克|kg|寸|英寸|小时|分钟|秒|%|倍|万|千|百万|亿)', text)
    for nu in num_units:
        words.append(nu)
    counter = Counter(words)
    keywords = [w for w, c in counter.items() if c >= 2]
    if len(keywords) < 5:
        keywords += [w for w, c in counter.items() if c == 1 and len(w) >= 3 and not w.isdigit() and w not in STOPWORDS]
    return list(dict.fromkeys(keywords))[:top_n]

def get_keywords_for_product(category, tier, lang='CN'):
    try:
        content = product_db[category][tier][lang]
    except KeyError:
        return []
    full_text = ""
    for fmt in ['List', 'FAQ', 'Paragraph']:
        if fmt in content:
            full_text += content[fmt] + "\n"
    return extract_short_keywords(full_text)

# Tier inference
price_map = {
    ('Apparel', 'budget'): 268, ('Apparel', 'medium'): 589, ('Apparel', 'premium'): 749,
    ('Cosmetics', 'budget'): 249, ('Cosmetics', 'medium'): 479, ('Cosmetics', 'premium'): 1400,
    ('Electronics', 'budget'): 2039, ('Electronics', 'medium'): 5499, ('Electronics', 'premium'): 9999,
}
brand_map = {
    ('Apparel', 'budget'): r'Champion|冠军',
    ('Apparel', 'medium'): r'Nike|耐克',
    ('Apparel', 'premium'): r'Adidas|阿迪达斯',
    ('Cosmetics', 'budget'): r'欧莱雅|L[-\']?Oreal',
    ('Cosmetics', 'medium'): r'兰蔻|Lanc[^a-z]?[oô]me',
    ('Cosmetics', 'premium'): r'香奈儿|Chanel',
    ('Electronics', 'budget'): r'OPPO',
    ('Electronics', 'medium'): r'Xiaomi|小米',
    ('Electronics', 'premium'): r'iPhone|Apple|苹果',
}
model_keywords = {
    ('Apparel', 'budget'): ['CHAMPRACER', '25SSR15'],
    ('Apparel', 'medium'): ['IB2764', 'Air Zoom Upturn'],
    ('Apparel', 'premium'): ['BOSTON 13', 'ADIZERO'],
    ('Cosmetics', 'budget'): ['小金牌', 'Golden Lift'],
    ('Cosmetics', 'medium'): ['持妆粉底液', 'Longwear'],
    ('Cosmetics', 'premium'): ['金砖', 'Sublimage'],
    ('Electronics', 'budget'): ['Reno14'],
    ('Electronics', 'medium'): ['17 Pro Max', 'Xiaomi 17'],
    ('Electronics', 'premium'): ['17 Pro Max', 'iPhone 17'],
}

def infer_tier(category, response_text):
    if not response_text:
        return None
    for (cat, tier), price in price_map.items():
        if cat == category and str(price) in response_text:
            return tier
    for (cat, tier), pattern in brand_map.items():
        if cat == category and re.search(pattern, response_text, re.IGNORECASE):
            return tier
    for (cat, tier), kws in model_keywords.items():
        if cat == category:
            for kw in kws:
                if kw.lower() in response_text.lower():
                    return tier
    return None

# ================== 2. Read CSV files ==================
csv_files = ["DeepSeek Results_Sorted.csv", "Kimi Results_Sorted.csv", "Qwen Results_Sorted.csv"]
all_data = []
for f in csv_files:
    if not os.path.exists(f):
        print(f"Warning: {f} not found, skipping")
        continue
    try:
        df = pd.read_csv(f, encoding='utf-8')
    except:
        df = pd.read_csv(f, encoding='gbk')
    print(f"Loaded {f}, rows: {len(df)}")
    all_data.append(df)

df_all = pd.concat(all_data, ignore_index=True)
df_all.columns = [col.strip() for col in df_all.columns]
rename_dict = {}
for col in df_all.columns:
    low = col.lower()
    if low == 'model': rename_dict[col] = 'Model'
    elif low == 'category': rename_dict[col] = 'Category'
    elif low == 'language': rename_dict[col] = 'Language'
    elif low == 'format': rename_dict[col] = 'Format'
    elif low == 'guide_type': rename_dict[col] = 'Guide_Type'
    elif low == 'run_id': rename_dict[col] = 'Run_ID'
    elif low == 'response': rename_dict[col] = 'Response'
df_all.rename(columns=rename_dict, inplace=True)
df_all = df_all[df_all['Language'].str.upper() == 'CN'].copy()

# ================== 3. Compute per-response metrics ==================
keywords_cache = {}
for cat in ['Apparel', 'Cosmetics', 'Electronics']:
    for tier in ['budget', 'medium', 'premium']:
        keywords_cache[(cat, tier)] = get_keywords_for_product(cat, tier, 'CN')
        print(f"{cat}-{tier} keywords: {keywords_cache[(cat, tier)]}")

records = []
skipped = 0
for idx, row in df_all.iterrows():
    cat = row['Category']
    resp = str(row['Response']).strip()
    if not resp:
        skipped += 1
        continue
    tier = infer_tier(cat, resp)
    if not tier:
        skipped += 1
        continue
    keywords = keywords_cache.get((cat, tier), [])
    if not keywords:
        skipped += 1
        continue
    resp_lower = resp.lower()
    total_kp = len(keywords)
    hit_count = 0
    pos_sum = 0.0
    for kw in keywords:
        kw_low = kw.lower()
        if kw_low in resp_lower:
            hit_count += 1
            pos = resp_lower.find(kw_low)
            pos_score = 1 - (pos / len(resp)) if len(resp) > 0 else 0
            pos_sum += pos_score
    retention = hit_count / total_kp if total_kp > 0 else 0.0
    avg_pos = pos_sum / total_kp if total_kp > 0 else 0.0
    records.append({
        'Model': row['Model'],
        'Category': cat,
        'Tier': tier,
        'Format': row['Format'],
        'Guide_Type': row.get('Guide_Type', ''),
        'Run_ID': row.get('Run_ID', idx),
        'Retention_Rate': retention,
        'Avg_Position_Score': avg_pos,
    })

result_df = pd.DataFrame(records)
print(f"Valid samples: {len(result_df)}, Skipped: {skipped}")

# ================== 4. Aggregate ==================
summary = result_df.groupby(['Model', 'Format']).agg(
    Citation_Freq=('Retention_Rate', 'mean'),
    Position_Weight=('Avg_Position_Score', 'mean'),
).reset_index()

# Preference score (normalized citation frequency)
model_global_freq = summary.groupby('Model')['Citation_Freq'].mean().to_dict()
summary['Preference_Score'] = summary.apply(
    lambda r: r['Citation_Freq'] / model_global_freq[r['Model']] if model_global_freq[r['Model']] > 0 else 1, axis=1
)

# Output matrices
pivot_freq = summary.pivot(index='Model', columns='Format', values='Citation_Freq')
pivot_pref = summary.pivot(index='Model', columns='Format', values='Preference_Score')
print("\n=== Citation Frequency ===")
print(pivot_freq.round(4))
print("\n=== Structural Preference Score (>1 preferred) ===")
print(pivot_pref.round(4))

# ================== 5. ANOVA ==================
result_df['Format'] = result_df['Format'].astype('category')
result_df['Model'] = result_df['Model'].astype('category')
model_anova = ols('Retention_Rate ~ C(Format) + C(Model) + C(Format):C(Model)', data=result_df).fit()
anova_table = sm.stats.anova_lm(model_anova, typ=2)
print("\n=== Two-way ANOVA (Format * Model) ===")
print(anova_table)
for effect in anova_table.index:
    p = anova_table.loc[effect, 'PR(>F)']
    sig = "significant" if p < 0.05 else "not significant"
    print(f"Effect '{effect}': p = {p:.5f} ({sig})")

# ================== 6. Final preference ==================
print("\n=== Final Preference ===")
pref_list = []
for m in summary['Model'].unique():
    sub = summary[summary['Model'] == m]
    best = sub.loc[sub['Preference_Score'].idxmax()]
    print(f"- {m}: most prefers {best['Format']} (pref score={best['Preference_Score']:.3f}, citation freq={best['Citation_Freq']:.3f})")
    pref_list.append({
        'Model': m,
        'Most_Preferred_Format': best['Format'],
        'Preference_Score': best['Preference_Score'],
        'Citation_Freq': best['Citation_Freq']
    })
pref_df = pd.DataFrame(pref_list)

# ================== 7. Generate 3 plots ==================
sns.set_theme(style='whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# Plot 1: Radar chart
formats = ['FAQ', 'List', 'Paragraph']
radar_data = []
for m in summary['Model'].unique():
    row = []
    for f in formats:
        val = summary[(summary['Model']==m) & (summary['Format']==f)]['Preference_Score'].values
        row.append(val[0] if len(val)>0 else 1)
    radar_data.append(row)

angles = [n / float(len(formats)) * 2 * pi for n in range(len(formats))]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
for i, m in enumerate(summary['Model'].unique()):
    values = radar_data[i]
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=m)
    ax.fill(angles, values, alpha=0.1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(formats)
ax.set_ylabel('Preference Score', fontsize=10)
ax.set_title('Preference Profile of Each Model for Content Formats', fontsize=14, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
ax.set_ylim(0, 1.5)
plt.tight_layout()
plt.savefig('preference_radar.png', dpi=300)
plt.show()

# Plot 2: GEO Index bar plot
summary['GEO_Index'] = 0.7 * summary['Citation_Freq'] + 0.3 * summary['Position_Weight']
plt.figure(figsize=(10,6))
sns.barplot(data=summary, x='Model', y='GEO_Index', hue='Format', palette='Set2')
plt.title('GEO Effectiveness Index')
plt.ylabel('GEO Index Value')
plt.tight_layout()
plt.savefig('geo_index_barplot.png', dpi=300)
plt.show()

# Plot 3: Normalized heatmap
min_val = summary['GEO_Index'].min()
max_val = summary['GEO_Index'].max()
summary['GEO_Norm'] = (summary['GEO_Index'] - min_val) / (max_val - min_val)
geo_norm_pivot = summary.pivot(index='Model', columns='Format', values='GEO_Norm')
plt.figure(figsize=(8,5))
sns.heatmap(geo_norm_pivot, annot=True, cmap='YlOrRd', fmt='.3f', linewidths=0.5,
            cbar_kws={'label': 'Normalized GEO Index'})
plt.title('GEO Effectiveness Index (Normalized)')
plt.tight_layout()
plt.savefig('geo_index_norm_heatmap.png', dpi=300)
plt.show()

print("\nSaved 3 plots: preference_radar.png, geo_index_barplot.png, geo_index_norm_heatmap.png")

# ================== 8. Save results to Excel ==================
output_file = 'GEO_results_summary.xlsx'
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    pivot_freq.to_excel(writer, sheet_name='Citation_Frequency')
    pivot_pref.to_excel(writer, sheet_name='Preference_Score')
    anova_table.to_excel(writer, sheet_name='ANOVA')
    pref_df.to_excel(writer, sheet_name='Final_Preference', index=False)
print(f"\nResults saved to {output_file}")

# Optionally also save as CSV files (uncomment if needed)
# pivot_freq.to_csv('Citation_Frequency.csv')
# pivot_pref.to_csv('Preference_Score.csv')
# anova_table.to_csv('ANOVA.csv')
# pref_df.to_csv('Final_Preference.csv', index=False)