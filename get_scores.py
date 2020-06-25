from requests_html import HTMLSession
import pandas as pd
import time


def main():
    session = HTMLSession()
    html = 'https://www.bbc.com/sport/football/premier-league/scores-fixtures/'
    month = ['2019-08', '2019-09', '2019-10', '2019-11', '2019-12', '2020-01', '2020-02', '2020-03']
    team = []
    score = []
    for m in month:
        tmp_team, tmp_score = [], []
        print('Start getting score of ', m)
        r = session.get(html + m)
        r_list = []
        for i in r.html.find('li'):
            if 'gs-o-list-ui__item' in i.attrs.get('class', []):
                r_list.append(i)

        for ele in r_list:
            tmp_team.append(ele.find('abbr')[0].attrs.get('title'))
            tmp_team.append(ele.find('abbr')[1].attrs.get('title'))

        for i in r.html.find('[class~=sp-c-fixture__number--ft]'):
            tmp_score.append(int(i.text))

        print('Get {} team and {} scores'.format(len(tmp_team), len(tmp_score)))
        print()
        team += tmp_team
        score += tmp_score
        time.sleep(3)

    print(len(team))
    print(len(score))

    res = pd.DataFrame({'home': team[::2], 'home_score': score[::2], 'away': team[1::2], 'away_score': score[1::2]})
    res.to_csv('output.csv')


main()
