import pandas as pd
import numpy as np

all_event_types = ['ADD_CART', 'VIEW', 'REMOVE_CART', 'BUY']

cat_cols = ["event_type", "product_id", "category_id", "user_id"]


def parse_dates(df):
    df = df.copy()
    df['event_time'] = pd.to_datetime(df['event_time'])
    df['event_date'] = df['event_time'].dt.date
    df['day_of_week'] = df['event_time'].dt.dayofweek
    df['hour'] = df['event_time'].dt.hour
    return df


def calculate_total_counts(df, id_col):
    df_counts = (
        df.pivot_table(
            index=id_col, columns='event_type', aggfunc='size', fill_value=0
        )
        .reindex(columns=all_event_types, fill_value=0)
        .reset_index()
    )
    prefix = id_col.split('_')[0]
    df_counts.columns = [id_col] + [f"{prefix}_{et.lower()}_count" for et in all_event_types]
    return df_counts


def calculate_session_features(df):
    df = df.copy()
    df.sort_values(by=['user_session', 'event_time'], inplace=True)
    session_start_time = df.groupby('user_session')['event_time'].transform('min')
    session_end_time = df.groupby('user_session')['event_time'].transform('max')

    df['time_since_session_start'] = (df['event_time'] - session_start_time).dt.total_seconds()
    df['session_event_count'] = df.groupby('user_session')['event_time'].transform('count')
    df['session_product_count'] = df.groupby('user_session')['product_id'].transform('nunique')
    df['session_category_count'] = df.groupby('user_session')['category_id'].transform('nunique')
    df['time_to_next_event'] = df.groupby('user_session')['event_time'].diff(periods=-1).dt.total_seconds().abs()
    df['is_last_event_of_session'] = (df['event_time'] == session_end_time).astype(int)
    df['session_duration'] = (session_end_time - session_start_time).dt.total_seconds()

    first_add_cart_time = df[df['event_type'] == 'ADD_CART'].groupby('user_session')['event_time'].transform('min')
    first_buy_time = df[df['event_type'] == 'BUY'].groupby('user_session')['event_time'].transform('min')
    df['time_to_first_add_cart'] = (first_add_cart_time - session_start_time).dt.total_seconds()
    df['time_to_first_buy'] = (first_buy_time - session_start_time).dt.total_seconds()

    df['session_unique_days'] = df.groupby('user_session')['event_time'].transform(lambda x: x.dt.date.nunique())
    df['session_day_span'] = df.groupby('user_session')['event_time'].transform(lambda x: (x.max().date() - x.min().date()).days)
    df['session_duration_minutes'] = df['session_duration'] / 60
    df['session_daily_avg_events'] = df['session_event_count'] / df['session_unique_days'].replace(0, 1)

    df['morning_event'] = df['event_time'].dt.hour.between(6, 11).astype(int)
    df['afternoon_event'] = df['event_time'].dt.hour.between(12, 17).astype(int)
    df['evening_event'] = df['event_time'].dt.hour.between(18, 23).astype(int)
    df['night_event'] = df['event_time'].dt.hour.between(0, 5).astype(int)

    inter_event_diff = df.groupby('user_session')['event_time'].diff().dt.total_seconds()
    df['mean_inter_event_sec'] = inter_event_diff.groupby(df['user_session']).transform('mean').fillna(0)
    df['std_inter_event_sec'] = inter_event_diff.groupby(df['user_session']).transform('std').fillna(0)

    return df


def add_daily_and_session_to_daily_features(df):
    df = df.copy()
    daily_stats = df.groupby('event_date').agg(
        daily_event_count=('event_time', 'count'),
        daily_unique_users=('user_id', 'nunique'),
        daily_unique_products=('product_id', 'nunique')
    ).reset_index()

    daily_event_counts = df.groupby(['event_date', 'event_type']).size().unstack(fill_value=0).reindex(columns=all_event_types, fill_value=0)
    daily_event_counts.columns = [f"daily_{col.lower()}_count" for col in daily_event_counts.columns]

    df = pd.merge(df, daily_stats, on='event_date', how='left')
    df = pd.merge(df, daily_event_counts.reset_index(), on='event_date', how='left')

    for etype in all_event_types:
        df[f'daily_{etype.lower()}_rate'] = df[f'daily_{etype.lower()}_count'] / df['daily_event_count']

    session_event_counts = df.groupby(['user_session', 'event_type']).size().unstack(fill_value=0).reindex(columns=all_event_types, fill_value=0).reset_index()
    df = pd.merge(df, session_event_counts, on='user_session', how='left')
    for etype in all_event_types:
        df[f'session_to_daily_{etype.lower()}_ratio'] = df[etype] / df[f'daily_{etype.lower()}_count']

    df.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    return df


def calculate_comprehensive_user_features(df):
    user_agg_df = df.groupby('user_id').agg(
        user_total_events=('event_type', 'count'),
        user_unique_sessions=('user_session', 'nunique'),
        user_unique_products=('product_id', 'nunique'),
        user_unique_categories=('category_id', 'nunique'),
        user_min_event_time=('event_time', 'min'),
        user_max_event_time=('event_time', 'max'),
        avg_time_between_events=('event_time', lambda x: x.diff().mean().total_seconds()),
        std_time_between_events=('event_time', lambda x: x.diff().std().total_seconds())
    ).reset_index()

    user_agg_df['user_total_duration'] = (user_agg_df['user_max_event_time'] - user_agg_df['user_min_event_time']).dt.total_seconds()

    user_unique_days = df.groupby('user_id')['event_time'].apply(lambda x: x.dt.date.nunique()).reset_index(name='user_unique_days')
    user_avg_events_per_day = df.groupby('user_id')['event_time'].apply(lambda x: x.count() / x.dt.date.nunique()).reset_index(name='user_avg_events_per_day')
    user_most_active_day = df.groupby('user_id')['event_time'].apply(lambda x: x.dt.dayofweek.value_counts().idxmax()).reset_index(name='user_most_active_day')
    user_days_range = df.groupby('user_id')['event_time'].apply(lambda x: (x.max().date() - x.min().date()).days).reset_index(name='user_days_range')

    user_agg_df = user_agg_df.merge(user_unique_days, on='user_id')
    user_agg_df = user_agg_df.merge(user_avg_events_per_day, on='user_id')
    user_agg_df = user_agg_df.merge(user_most_active_day, on='user_id')
    user_agg_df = user_agg_df.merge(user_days_range, on='user_id')

    user_event_counts = df.groupby(['user_id', 'event_type']).size().unstack(fill_value=0).reindex(columns=all_event_types, fill_value=0).reset_index()
    user_event_counts.columns = ['user_id'] + [f"user_{col.lower()}_count" for col in all_event_types]
    user_agg_df = user_agg_df.merge(user_event_counts, on='user_id')

    for etype in all_event_types:
        user_agg_df[f'user_{etype.lower()}_rate'] = user_agg_df[f'user_{etype.lower()}_count'] / user_agg_df['user_total_events']

    user_agg_df['user_view_to_add_cart_ratio'] = user_agg_df['user_view_count'] / user_agg_df['user_add_cart_count'].replace(0, 1)
    user_agg_df['user_add_cart_to_buy_ratio'] = user_agg_df['user_add_cart_count'] / user_agg_df['user_buy_count'].replace(0, 1)
    user_agg_df['user_buy_to_total_ratio'] = user_agg_df['user_buy_count'] / user_agg_df['user_total_events']
    user_agg_df['user_cart_abandon_ratio'] = (user_agg_df['user_add_cart_count'] - user_agg_df['user_buy_count']) / user_agg_df['user_add_cart_count'].replace(0, 1)

    user_agg_df.rename(columns={
        'avg_time_between_events': 'user_avg_time_between_events',
        'std_time_between_events': 'user_std_time_between_events'
    }, inplace=True)

    user_agg_df.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    user_agg_df = user_agg_df.merge(
        df.groupby('user_id')['session_duration'].mean().reset_index(name='user_avg_session_duration'),
        on='user_id'
    )
    return user_agg_df


def calculate_comprehensive_product_features(df):
    product_agg_df = df.groupby('product_id').agg(
        product_total_events=('event_type', 'count'),
        product_unique_users=('user_id', 'nunique'),
        product_unique_sessions=('user_session', 'nunique'),
        product_unique_categories=('category_id', 'nunique'),
        product_min_event_time=('event_time', 'min'),
        product_max_event_time=('event_time', 'max'),
        product_avg_time_between_events=('event_time', lambda x: x.diff().mean().total_seconds()),
        product_std_time_between_events=('event_time', lambda x: x.diff().std().total_seconds())
    ).reset_index()

    product_agg_df['product_total_duration'] = (
        product_agg_df['product_max_event_time'] - product_agg_df['product_min_event_time']
    ).dt.total_seconds()
    product_agg_df.drop(columns=['product_min_event_time', 'product_max_event_time'], inplace=True)

    product_event_counts = df.groupby(['product_id', 'event_type']).size().unstack(fill_value=0).reindex(columns=all_event_types, fill_value=0).reset_index()
    product_event_counts.columns = ['product_id'] + [f"product_{col.lower()}_count" for col in all_event_types]
    product_agg_df = product_agg_df.merge(product_event_counts, on='product_id')

    for etype in all_event_types:
        product_agg_df[f'product_{etype.lower()}_rate'] = product_agg_df[f'product_{etype.lower()}_count'] / product_agg_df['product_total_events']

    product_agg_df['product_view_to_add_cart_ratio'] = product_agg_df['product_view_count'] / product_agg_df['product_add_cart_count'].replace(0, 1)
    product_agg_df['product_add_cart_to_buy_ratio'] = product_agg_df['product_add_cart_count'] / product_agg_df['product_buy_count'].replace(0, 1)
    product_agg_df['product_buy_to_total_ratio'] = product_agg_df['product_buy_count'] / product_agg_df['product_total_events']
    product_agg_df['product_cart_abandon_ratio'] = (
        (product_agg_df['product_add_cart_count'] - product_agg_df['product_buy_count']) / product_agg_df['product_add_cart_count'].replace(0, 1)
    )
    product_agg_df.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    return product_agg_df


def calculate_comprehensive_category_features(df):
    category_agg_df = df.groupby('category_id').agg(
        category_total_events=('event_type', 'count'),
        category_unique_users=('user_id', 'nunique'),
        category_unique_sessions=('user_session', 'nunique'),
        category_unique_products=('product_id', 'nunique'),
        category_min_event_time=('event_time', 'min'),
        category_max_event_time=('event_time', 'max'),
        category_avg_time_between_events=('event_time', lambda x: x.diff().mean().total_seconds()),
        category_std_time_between_events=('event_time', lambda x: x.diff().std().total_seconds())
    ).reset_index()

    category_agg_df['category_total_duration'] = (
        category_agg_df['category_max_event_time'] - category_agg_df['category_min_event_time']
    ).dt.total_seconds()
    category_agg_df.drop(columns=['category_min_event_time', 'category_max_event_time'], inplace=True)

    category_event_counts = df.groupby(['category_id', 'event_type']).size().unstack(fill_value=0).reindex(columns=all_event_types, fill_value=0).reset_index()
    category_event_counts.columns = ['category_id'] + [f"category_{col.lower()}_count" for col in all_event_types]
    category_agg_df = category_agg_df.merge(category_event_counts, on='category_id')

    for etype in all_event_types:
        category_agg_df[f'category_{etype.lower()}_rate'] = category_agg_df[f'category_{etype.lower()}_count'] / category_agg_df['category_total_events']

    category_agg_df['category_view_to_add_cart_ratio'] = category_agg_df['category_view_count'] / category_agg_df['category_add_cart_count'].replace(0, 1)
    category_agg_df['category_add_cart_to_buy_ratio'] = category_agg_df['category_add_cart_count'] / category_agg_df['category_buy_count'].replace(0, 1)
    category_agg_df['category_buy_to_total_ratio'] = category_agg_df['category_buy_count'] / category_agg_df['category_total_events']
    category_agg_df['category_cart_abandon_ratio'] = (
        (category_agg_df['category_add_cart_count'] - category_agg_df['category_buy_count']) / category_agg_df['category_add_cart_count'].replace(0, 1)
    )
    category_agg_df.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    return category_agg_df


def add_session_event_features(df):
    session_event_counts = df.groupby(['user_session', 'event_type']).size().unstack(fill_value=0).reindex(columns=all_event_types, fill_value=0)
    session_event_counts['total_events'] = session_event_counts.sum(axis=1)
    session_event_counts['mean_events'] = session_event_counts.mean(axis=1)
    session_event_counts['std_events'] = session_event_counts.std(axis=1)
    session_event_counts['max_events'] = session_event_counts.max(axis=1)
    session_event_counts['min_events'] = session_event_counts.min(axis=1)

    for etype in all_event_types:
        session_event_counts[f'{etype.lower()}_rate'] = session_event_counts[etype] / session_event_counts['total_events']

    session_event_counts['view_to_add_cart_ratio'] = session_event_counts['VIEW'] / session_event_counts['ADD_CART'].replace(0, 1)
    session_event_counts['view_to_remove_cart_ratio'] = session_event_counts['VIEW'] / session_event_counts['REMOVE_CART'].replace(0, 1)
    session_event_counts['add_cart_to_buy_ratio'] = session_event_counts['ADD_CART'] / session_event_counts['BUY'].replace(0, 1)
    session_event_counts['add_cart_to_remove_cart_ratio'] = session_event_counts['ADD_CART'] / session_event_counts['REMOVE_CART'].replace(0, 1)
    session_event_counts.replace([np.inf, -np.inf, np.nan], 0, inplace=True)

    session_times = df.groupby('user_session')['event_time']
    session_event_counts['session_duration_sec'] = (session_times.max() - session_times.min()).dt.total_seconds()
    session_event_counts['mean_inter_event_sec_y'] = session_times.apply(lambda x: x.diff().dt.total_seconds().mean()).fillna(0)
    session_event_counts['std_inter_event_sec_y'] = session_times.apply(lambda x: x.diff().dt.total_seconds().std()).fillna(0)
    session_event_counts['unique_days'] = session_times.apply(lambda x: x.dt.date.nunique())
    session_event_counts['morning_events'] = df[df['event_time'].dt.hour.between(6, 11)].groupby('user_session')['event_time'].count()
    session_event_counts['afternoon_events'] = df[df['event_time'].dt.hour.between(12, 17)].groupby('user_session')['event_time'].count()
    session_event_counts['evening_events'] = df[df['event_time'].dt.hour.between(18, 23)].groupby('user_session')['event_time'].count()
    session_event_counts['night_events'] = df[df['event_time'].dt.hour.between(0, 5)].groupby('user_session')['event_time'].count()
    session_event_counts.fillna(0, inplace=True)
    session_event_counts['unique_products'] = df.groupby('user_session')['product_id'].nunique()
    session_event_counts['unique_categories'] = df.groupby('user_session')['category_id'].nunique()
    return pd.merge(df, session_event_counts.reset_index(), on='user_session', how='left')


def engineer_features(df, is_train=True):
    df = parse_dates(df)
    df = calculate_session_features(df)

    user_counts = calculate_total_counts(df, 'user_id')
    category_counts = calculate_total_counts(df, 'category_id')
    product_counts = calculate_total_counts(df, 'product_id')

    user_comprehensive = calculate_comprehensive_user_features(df)
    product_comprehensive = calculate_comprehensive_product_features(df)
    category_comprehensive = calculate_comprehensive_category_features(df)

    df = df.merge(user_counts, on='user_id')
    df = df.merge(category_counts, on='category_id')
    df = df.merge(product_counts, on='product_id')
    df = add_daily_and_session_to_daily_features(df)
    df = df.merge(user_comprehensive, on='user_id')
    df = df.merge(product_comprehensive, on='product_id')
    df = df.merge(category_comprehensive, on='category_id')
    df = add_session_event_features(df)
    return df


def get_feature_columns(df):
    numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    for drop in ['event_time', 'session_value', 'pred']:
        if drop in numeric_cols:
            numeric_cols.remove(drop)
    return numeric_cols + cat_cols
