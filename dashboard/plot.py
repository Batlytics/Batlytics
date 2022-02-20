import pandas as pd
import plotly.express as px

df = pd.read_csv('../dataset/02_02_21_Scrimmage.csv', low_memory=False)


def trajectory_xc0_yc0_zc0(df):
    new_df = df[['PitchTrajectoryXc0', 'PitchTrajectoryYc0',
                 'PitchTrajectoryZc0', 'AutoPitchType']]
    fig = px.scatter_3d(
        z=new_df['PitchTrajectoryXc0'],
        x=new_df['PitchTrajectoryYc0'],
        y=new_df['PitchTrajectoryZc0'],
        color=new_df['AutoPitchType']
    )
    fig.update_layout(
        title='Ball Trajectory for Xc1, Yc1 and Zc1',
        width=800
    )
    return fig


# helper function to calculate points
def calculate_ploynomial(df):
    df_poly = {
        'x': [],
        'y': [],
        'z': [],
        'TaggedPitchType': [],
        'AutoPitchType': [],
        'PitchCall': [],
        'PlayResult': [],
        'RelSpeed': [],
        'SpinRate': [],
        'SpinAxis' : [], 
        'ExitSpeed' : []
    }
    for index, row in df.iterrows():
        x_data = row['PitchTrajectoryXc2'] - \
            (row['PitchTrajectoryXc1'] + row['PitchTrajectoryXc0'])
        y_data = row['PitchTrajectoryYc2'] - \
            (row['PitchTrajectoryYc1'] + row['PitchTrajectoryYc0'])
        z_data = row['PitchTrajectoryZc2'] - \
            (row['PitchTrajectoryZc1'] + row['PitchTrajectoryZc0'])
        tpt_data = row['TaggedPitchType']
        apt_data = row['AutoPitchType']
        pc_data = row['PitchCall']
        pr_data = row['PlayResult']
        rs_data = row['RelSpeed']
        sr_data = row['SpinRate']
        sa_data = row['SpinAxis']
        es_data = row['ExitSpeed']

        df_poly['x'].append(x_data)
        df_poly['y'].append(y_data)
        df_poly['z'].append(z_data)
        df_poly['TaggedPitchType'].append(tpt_data)
        df_poly['AutoPitchType'].append(apt_data)
        df_poly['PitchCall'].append(pc_data)
        df_poly['PlayResult'].append(pr_data)
        df_poly['RelSpeed'].append(rs_data)
        df_poly['SpinRate'].append(sr_data)
        df_poly['SpinAxis'].append(sa_data)
        df_poly['ExitSpeed'].append(es_data)
    return df_poly


def overall_pitch_trajectory(df, color):
    # polynomial
    df_polinomial = df[
        [
            'PitchTrajectoryXc0', 'PitchTrajectoryXc1', 'PitchTrajectoryXc2',
            'PitchTrajectoryYc0', 'PitchTrajectoryYc1', 'PitchTrajectoryYc2',
            'PitchTrajectoryZc0', 'PitchTrajectoryZc1', 'PitchTrajectoryZc2',
            'TaggedPitchType', 'AutoPitchType', 'PitchCall', 'PlayResult',
            'RelSpeed', 'SpinRate', 'SpinAxis', 'ExitSpeed'
        ]
    ]
    new_ploy_data = calculate_ploynomial(df_polinomial)
    poly_df = pd.DataFrame(
        new_ploy_data,
        columns=[
            'x',
            'y',
            'z',
            'TaggedPitchType',
            'AutoPitchType',
            'PitchCall',
            'PlayResult',
            'RelSpeed', 
            'SpinRate', 
            'SpinAxis', 
            'ExitSpeed'
        ]
    )
    fig = px.scatter_3d(
        x=poly_df['x'],
        y=poly_df['y'],
        z=poly_df['z'],
        color=poly_df[color]
    )

    fig.update_layout(
        title='Overall Pitch Trajectory movements',
        width=800
    )
    return fig

def speed_metric(df):
    # Avg relative speed(miles per hour)
    sum_speed = sum(df['RelSpeed'])
    avg_speed = sum_speed/(len(df['RelSpeed']))
    return round(avg_speed, 1)

def spinAngle_metric(df):
    # Avg ball spinAxis
    sum_spin = sum(df['SpinAxis'])
    avg_spin = sum_spin/(len(df['SpinAxis']))
    return round(avg_spin, 1)
