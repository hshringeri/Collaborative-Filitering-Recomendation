import pandas as pd
import numpy as np
from scipy.spatial import distance
import math


def makeDf(csvFile):
    df = pd.read_csv(csvFile)
    return df

def hi():
    df = makeDf("CS473_ResturantsRatings.csv")
    df = df.drop(['ID', 'Start time', 'Completion time'], axis=1)
    list = []
    for rowIndex, row in df.iterrows():  # iterate over rows
        count = 0
        total = 0
        for columnIndex, value in row.items():
            bool = np.isnan(value)
            if not bool:
                total += int(value)
                count += 1
        avg = total / count
        list.append([rowIndex, avg])


    list = sorted(list, key=lambda x: x[1], reverse=True)
    print(list)

def get_ovr_ratings():
    df = makeDf("CS473_ResturantsRatings.csv")
    df = df.drop(['ID', 'Start time', 'Completion time'], axis=1)
    count = 0
    total = 0
    for rowIndex, row in df.iterrows():  # iterate over rows
        for columnIndex, value in row.items():
            bool = np.isnan(value)
            if not bool:
                total += int(value)
                count += 1
    avg = total / count

    print("avg")
    print(avg)
    return avg

def normalize(avg):
    df = makeDf("CS473_ResturantsRatings.csv")
    df = df.drop(['ID', 'Start time', 'Completion time'], axis=1)

    for rowIndex, row in df.iterrows():  # iterate over rows
        for columnIndex, value in row.items():
            bool = np.isnan(value)
            if bool:
                df.at[rowIndex, columnIndex] = 0
            else:
                df.at[rowIndex, columnIndex] = value - avg

    print(df)
    return df


def cos_sim():
    df = normalize(get_ovr_ratings())
    og_df = makeDf("CS473_ResturantsRatings.csv")
    arr = df.to_numpy()

    vect_dict = {}
    for rowIndex, row in df.iterrows():
        id = og_df.at[rowIndex, 'ID']
        vect_dict[id] = arr[rowIndex]
    cos_sims = []
    for key in vect_dict:
        if key != 44:
            vect = vect_dict[key]
            cos_sims.append([key, 1 - distance.cosine(vect_dict[44], vect)])

    cos_sims = sorted(cos_sims, key=lambda x: x[1], reverse=True)
    print("similar users")
    print(cos_sims)

    ## find similar items to Chef Bill Kim's
    cbk_sim = []
    for column in df:
        if column != 'Unnamed: 36':
            cbk_sim.append([column, 1 - distance.cosine(df['Chef Bill Kim’s'], df[column])])

    cbk_sim = sorted(cbk_sim, key=lambda x: x[1], reverse=True)
    print(cbk_sim)

    ## find similar items to Main Street Poke
    msp_sim = []
    for column in df:
        if column != 'Unnamed: 36':
            msp_sim.append([column, 1 - distance.cosine(df['Main Street Poké'], df[column])])

    msp_sim = sorted(msp_sim, key=lambda x: x[1], reverse=True)
    print(msp_sim)





# might have to change if b changes
def rating_pred():
    avg = get_ovr_ratings()
    df = normalize(avg)
    og_df = makeDf("CS473_ResturantsRatings.csv")
    new_df = pd.DataFrame(og_df['ID'])
    new_df['Chef Bill Kim’s'] = df['Chef Bill Kim’s']
    new_df['Main Street Poké'] = df['Main Street Poké']
    new_df.to_csv('/Users/hardh/PycharmProjects/CS473/hw5Q3/normalized.csv')

    # 3 closest: (16, 0.457), (23, 0.348), (6, 0.347) -> chef bills
    # 3 closest: (43, 0.362), (10, 0.354), (6, 0.347) -> main street poke

    #user-user for Chef Bill Kim's
    uu_pred_num = ((0.457 * -1.5322085889570600) + (0.348 * -0.5322085889570550) + (0.347 * -1.5322085889570600))
    uu_pred_den = 0.457 + 0.348 + 0.347
    print("bill")
    print(uu_pred_num/uu_pred_den + avg)

    # user-user for Main Street Poke
    uu_pred_num_msp = ((0.362 * -0.532208588957055) + (0.354 * -0.5322085889570550) + (0.347 * -0.5322085889570550))
    uu_pred_den_msp = (0.362 + 0.354 + 0.347)
    print("poke")
    print(uu_pred_num_msp/uu_pred_den_msp + avg)

    ##item-item
    df.insert(0, 'ID', og_df['ID'])
    df.to_csv('/Users/hardh/PycharmProjects/CS473/hw5Q3/full_normalized.csv')

    #2 closest to Chef Bill Kims: (BBQ District, 0.739), (Pizza Parm Shop, 0.615)
    ii_pred_num_cbk = (0.739 * -0.532208588957055) + (0.615 * 0.46779141104294500)
    ii_pred_den_cbk = 0.739 + 0.615
    print("ii- cbk")
    print(ii_pred_num_cbk/ii_pred_den_cbk + avg)

    #2 closest to Main Street Poke: (Sol Toro, 0.568), (Lavassa, 0.462)
    ii_pred_num_msp = (0.568 * -0.5322085889570550) + (0.462 * 0.46779141104294500)
    ii_pred_den_msp = 0.568 + 0.462
    print("ii- msp")
    print(ii_pred_num_msp / ii_pred_den_msp + avg)


    # 2 closest to Main Street Poke




if __name__ == '__main__':
    normalize(get_ovr_ratings())
    cos_sim()
    rating_pred()
    hi()
