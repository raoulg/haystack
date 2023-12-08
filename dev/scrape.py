import requests
from bs4 import BeautifulSoup
from loguru import logger
from tqdm import tqdm



class ThesisScraper:
    def __init__(self, base_url: str, search_url: str, datafolder: Path) -> None:
        self.base_url = base_url
        self.search_url = search_url
        self.datafolder = datafolder
        self.info = []

    def __call__(self) -> list:
        self.run()

    def download(self, max_downloads: int = 1000):
        for i in tqdm(range(max_downloads)):
            self.download_file(self.info[i])


    def __repr__(self) -> str:
        return f'ThesisScraper(base_url={self.base_url})'

    def run(self):
        session = requests.Session()
        info = []

        for i in tqdm(range(100)):
            current_url = self.base_url + self.search_url
            response = session.get(current_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            sublinks = soup.find_all('a', href=True)
            detail_links = self.get_details(sublinks)
            for detail_url in detail_links:
                download_url, features = self.get_subpage_info(detail_url, session)
                self.info.append([download_url, features])
            pagination = soup.find_all('a', class_='search__body__pagination__arrow')
            if 'is-disabled' not in pagination[1].attrs['class']:
                search_url = pagination[1]['href']
            else:
                logger.info(f'No more pages to scrape at page {i+1}.')
                break


    @staticmethod
    def get_details(subpage_links: list) -> list:
        detail_links = []
        for link in subpage_links:
            href = link['href']
            if href.startswith('/details'):
                detail_links.append(href)
        return detail_links

    def get_subpage_info(self, detail_url: str, session: requests.Session) -> tuple:
        response = session.get(self.base_url + detail_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        download_url = soup.find('a', class_='detail__header__button')
        features = self._obtain_features_from_table(soup)
        features["title"] = self._get_title(soup)

        if download_url and 'href' in download_url.attrs:
            download_url = self.base_url + download_url['href']

        return download_url, features

    @staticmethod
    def _get_title(soup) -> str:
        div_tag = soup.find('div', class_='detail__header__column')
        title = div_tag.find('h1').text.replace(' ', '_')
        return title

    def download_file(self, info):
        url = info[0]
        feat = info[1]
        filename = Path(feat["title"]).with_suffix('.pdf')
        subdir = Path(feat["organisatie"].replace(' ', '_')) / Path(feat["opleiding"].replace(' ', '_'))
        filepath = Path(self.datafolder) / subdir
        if not filepath.exists():
            filepath.mkdir(parents=True)
        response = session.get(url, stream=True)
        with (filepath / filename).open('wb') as f:
            for chunk in response.iter_content(chunk_size=128):
                f.write(chunk)

    @staticmethod
    def _obtain_features_from_table(soup) -> dict:
        rows = soup.find_all('tr')
        organisatie = None
        opleiding = None

        # Iterate through each row and check the label in the first cell
        for row in rows:
            cells = row.find_all('td')
            if len(cells) >= 2:  # Ensure there are at least two cells in the row
                label = cells[0].text.strip()
                value = cells[1].text.strip()

                # Check for 'Organisatie' and 'Opleiding' labels
                if label == 'Organisatie':
                    organisatie = value
                elif label == 'Opleiding':
                    opleiding = value
        return {"organisatie" : organisatie, "opleiding" : opleiding}

if __name__ == "__main__":
    base_url = "https://www.hbokennisbank.nl"
    search_url = "/searchresult?q=&sort-order=date&date-from=&date-until=&t-0-k=hbo%3Aproduct&t-0-v=info%3Aeu-repo%2Fsemantics%2FbachelorThesis&t-0-v=info%3Aeu-repo%2Fsemantics%2FmasterThesis&t-0-v=info%3Aeu-repo%2Fsemantics%2FassociateDegree&c=2&has-link=yes"
    scraper = ThesisScraper(base_url, search_url, datafolder=Path('downloads'))
    scraper()
    scraper.download(max_downloads=1000)