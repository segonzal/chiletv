import argh
from typing import List
from pathlib import Path

from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from spiders import LaTerceraSpider, ElMostradorSpider, EmolSpider, LaCuartaSpider


SPIDER_MAP = {
    'latercera': LaTerceraSpider,
    'elmostrador': ElMostradorSpider,
    'emol': EmolSpider,
    'lacuarta': LaCuartaSpider,
}
SPIDER_CHOICES = list(SPIDER_MAP.keys())


@argh.arg('dst', help='Data storage folder.')
@argh.arg('-s', '--spiders', nargs='+', type=str, help='Spiders to run.', choices=SPIDER_CHOICES)
def main(dst: str, spiders: List[str] = list):
    dst = Path(dst)
    spiders = [SPIDER_MAP[s] for s in spiders]
    settings = get_project_settings()
    settings.update({
        'FEEDS': {
            str(dst / '%(name)s.json'): {
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
    argh.dispatch_command(main)
