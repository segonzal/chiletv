from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from spiders.latercera import LaTerceraSpider
from spiders.elmostrador import ElMostradorSpider
from spiders.emol import EmolSpider
from spiders.lacuarta import LaCuartaSpider


def main():
    spiders = [
        # LaTerceraSpider,
        # ElMostradorSpider,
        # EmolSpider,
        LaCuartaSpider,
    ]
    settings = get_project_settings()
    settings.update({
        'FEEDS': {
            'news/%(name)s.json': {
                'format': 'jsonlines',
                'encoding': 'utf8',
                'store_empty': False,
                'overwrite': True,
            },
        },
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 5.1; rv:5.0) Gecko/20100101 Firefox/5.0',
        'DOWNLOAD_DELAY': 0.25,
        'LOG_LEVEL': 'INFO',
        'CLOSESPIDER_PAGECOUNT': 10000,
    })
    process = CrawlerProcess(settings)
    for spider in spiders:
        process.crawl(spider)
    process.start()


if __name__ == '__main__':
    main()
