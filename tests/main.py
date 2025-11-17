"""Entry point for testing"""

#from * as  import BeautifulSoup
#import requests

year: str = "2025"
url: str = "http://www.imdb.com/search/title?release_date="+year+","+year+"&title_type=feature"

#ourUrl = urllib3.PoolManager().request('GET', url).data
#soup = BeautifulSoup(url)
#soup = BeautifulSoup(ourUrl, "lxml")
#print(soup.find('title').text)


def main():
    """_summary_
    main
    """
    print('TESTING')
    #raise Exception('Test Exception')

main()
