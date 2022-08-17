import argh
from typing import List
from pathlib import Path

from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from spiders import LaTerceraSpider, ElMostradorSpider, EmolSpider, LaCuartaSpider, TheClinicSpider


SPIDER_MAP = {
    'latercera': LaTerceraSpider,
    'elmostrador': ElMostradorSpider,
    'emol': EmolSpider,
    'lacuarta': LaCuartaSpider,
    'theclinic': TheClinicSpider,
}
SPIDER_CHOICES = list(SPIDER_MAP.keys())


#@argh.arg('dst', type=Path, help='Data storage folder.')
@argh.arg('-s', '--spiders', nargs='+', type=str, default=SPIDER_CHOICES, help='Spiders to run.', choices=SPIDER_CHOICES)
@argh.arg('-l', '--loglevel', type=str, default='ERROR', help='Log level.', choices=['INFO', 'ERROR', 'DEBUG'])
def main(spiders: List[str] = list, loglevel: str = 'ERROR'):
    spiders = [SPIDER_MAP[s] for s in spiders]
    settings = get_project_settings()
    settings.update({
        'FEEDS': {
            'data/%(name)s_%(time)s.json': {
                'format': 'jsonlines',
                'encoding': 'utf8',
                'store_empty': False,
                'overwrite': True,
            },
        },
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 5.1; rv:5.0) Gecko/20100101 Firefox/5.0',
        'DOWNLOAD_DELAY': 0.25,
        'LOG_LEVEL': loglevel,
        # 'CLOSESPIDER_PAGECOUNT': 10000,
    })
    process = CrawlerProcess(settings)
    for spider in spiders:
        spider.custom_settings = {'JOBDIR': 'data/crawl-'+spider.name}
        process.crawl(spider)
    process.start()
    #process.join()
    #process.stop()


if __name__ == '__main__':
    argh.dispatch_command(main)
