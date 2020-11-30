import random


def rate(ratings, wave_list, fps, dead=False):
    for wave in wave_list:
        if len(wave.frame_data) >= fps or dead:
            for i in range(min(fps, len(wave.frame_data))):
                ratings.append(placeholder(wave.frame_data[i]))


def placeholder(wave_frame):
    return random.randint(1, 10)


def get_final_rating(ratings):
    top_five_per = []
    min = 0

    if len(ratings) == 0:
        print ("Error: no ratings to analyze")
        return 1

    if len(ratings) < 5:
        return max(ratings)

    tenth_percentile = len(ratings) / 5

    for r in ratings:
        if len(top_five_per) < tenth_percentile:
            top_five_per.append(r)
        else:
            for ttr in top_five_per:
                if r > ttr:
                    top_five_per.remove(ttr)
                    top_five_per.append(r)
                    break
    print(top_five_per)

    return sum(top_five_per) / len(top_five_per)
