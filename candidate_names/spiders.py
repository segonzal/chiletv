from datetime import datetime

from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule

from .utils import pick_longest, extract_content, repair_item, remove_nodes

class ElDinamoSpider(CrawlSpider):
    name = 'eldinamo'
    allowed_domains = ['eldinamo.cl']
    start_urls = ['https://www.eldinamo.cl/']

    rules = ()

    def parse_item(self, response):
        pass


class ElMostradorSpider(CrawlSpider):
    name = 'elmostrador'
    allowed_domains = ['elmostrador.cl']
    start_urls = ['https://www.elmostrador.cl/']

    rules = (
        Rule(LinkExtractor(allow=r'.*/\d{4}/\d{2}/\d{2}/.*', deny=[r'noticias/multimedia/.*',
                                                                   r'noticias/mundo/.*',
                                                                   r'noticias/sin-editar/.*',
                                                                   r'.*\.pdf$']),
             callback='parse_item', follow=True),
        Rule(LinkExtractor(allow=r'.*'), follow=True),
    )

    def parse_item(self, response):
        item = {}
        item['title'] = response.xpath('//meta[@property="og:title"]/@content').get()
        item['pubDate'] = datetime.strptime(
            response.xpath('//meta[@property="article:published_time"]/@content').get()[:19], '%Y-%m-%dT%H:%M:%S')
        item['category'] = set()
        item['category'].update(response.xpath('//meta[@property="article:tag"]/@content').getall())
        item['category'].update(response.xpath('//meta[@property="article:section"]/@content').getall())
        item['guid'] = response.xpath('//link[@rel="shortlink"]/@href').get()
        item['description'] = pick_longest(response.xpath('//meta[@property="og:description"]/@content').getall())
        item['content'] = extract_content(response.css('div#noticia > p, div#noticia > h3'))
        item['crawlDate'] = datetime.utcnow()
        return repair_item(item)


class EmolSpider(CrawlSpider):
    name = 'emol'
    allowed_domains = ['emol.com']
    start_urls = ['https://www.emol.com/']

    rules = (
        Rule(LinkExtractor(allow=[r'noticias/' + k + r'/\d{4}/\d{2}/\d{2}/\d+/.*' for k in [
            'Nacional', 'Internacional', 'Economia', 'Deportes', 'Espectaculos', 'Tendencias', 'Autos']]),
             callback='parse_item', follow=True),
        Rule(LinkExtractor(allow=r'.*'), follow=True),
    )

    def parse_item(self, response):
        title = response.xpath('//meta[@property="og:title"]/@content').get()
        pubDate = response.xpath('//meta[@property="article:published_time"]/@content').get()
        category = set()
        category.update(response.xpath('//meta[@property="article:tag"]/@content').getall())
        category.update(response.xpath('//meta[@property="article:section"]/@content').getall())
        guid = response.xpath('//meta[@property="og:url"]/@content').get()
        description = response.xpath('//meta[@property="og:description"]/@content').get()
        content = response.css('div#cuDetalle_cuTexto_textoNoticia')
        content = remove_nodes(content, ['div[id^="contRelacionada"]', 'script', 'div.contenedor_video_iframe'])

        if title:
            return repair_item({
                'title': title[:-len(' | Emol.com')],
                'pubDate': datetime.strptime(pubDate[:19], '%Y-%m-%dT%H:%M:%S'),
                'category': category,
                'guid': guid,
                'description': description,
                'content': extract_content(content),
                'crawlDate': datetime.utcnow(),
            })


class LaCuartaSpider(CrawlSpider):
    name = 'lacuarta'
    allowed_domains = ['lacuarta.com']
    start_urls = ['https://www.lacuarta.com/']

    rules = (
        Rule(LinkExtractor(allow=[k + r'/noticia/.*' for k in [
            'cronica', 'espectaculos', 'cronica', 'deportes', 'tendencias', 'servicios', 'el-faro']]),
             callback='parse_item', follow=True),
        Rule(LinkExtractor(allow=r'.*'), follow=True),
    )

    def parse_item(self, response):
        title = response.xpath('//meta[@property="og:title"]/@content').get()
        pubDate = response.css('div.story-content article time').attrib['datetime']
        category = set(response.css('div.noreadme-audima li>a::text').getall())
        guid = response.xpath('//meta[@property="og:url"]/@content').get()
        description = response.xpath('//meta[@property="og:description"]/@content').get()
        content = response.css('article section:first-child>:not(figure):not(div.container):not(div.story-twitter):not(div.story-instagram)')

        return repair_item({
            'title': title,
            'pubDate': datetime.strptime(pubDate[:19], '%Y-%m-%dT%H:%M:%S'),
            'category': category,
            'guid': guid,
            'description': description,
            'content': extract_content(content),
            'crawlDate': datetime.utcnow(),
        })


class LaNacionSpider(CrawlSpider):
    name = 'lanacion'
    allowed_domains = ['lanacion.cl']
    start_urls = ['http://www.lanacion.cl/']

    rules = ()

    def parse_item(self, response):
        pass


class LaRazonSpider(CrawlSpider):
    name = 'larazon'
    allowed_domains = ['lanrazon.cl']
    start_urls = ['https://www.larazon.cl/']

    rules = ()

    def parse_item(self, response):
        pass


class LaTerceraSpider(CrawlSpider):
    name = 'latercera'
    allowed_domains = ['latercera.com']
    start_urls = ['https://www.latercera.com/']

    rules = (
        Rule(LinkExtractor(allow=[k + r'/noticia/.*' for k in [
            'earlyaccess', 'pulso', 'nacional', 'reconstitucion', 'laboratoriodecontenidos', 'que-pasa', 'el-deportivo',
            'opinion', 'culto', 'paula', 'pulso-pm'
        ]]),
             callback='parse_item', follow=True),
        Rule(LinkExtractor(allow=r'.*'), follow=True),
    )

    def parse_item(self, response):
        pubdate = response.xpath('//meta[@property="article:published_time"]/@content').get()

        if pubdate:
            item = {}
            item['title'] = pick_longest(response.xpath('//meta[@property="og:title"]/@content').getall())[:-13]
            item['pubDate'] = datetime.strptime(pubdate[:19], '%Y-%m-%dT%H:%M:%S')
            item['category'] = set()
            item['category'].update(response.xpath('//meta[@property="article:tag"]/@content').getall())
            item['category'].update(response.xpath('//meta[@property="article:section"]/@content').getall())
            item['guid'] = response.xpath('//meta[@property="og:url"]/@content').get()
            item['description'] = pick_longest(response.xpath('//meta[@property="og:description"]/@content').getall())
            item['content'] = extract_content(response.css('article div.single-content > p, div.header'))
            item['crawlDate'] = datetime.utcnow()
            return repair_item(item)


class LaVozDeLosQueSobranSpider(CrawlSpider):
    name = 'lavozdelosquesobran'
    allowed_domains = ['lavozdelosquesobran.cl']
    start_urls = ['https://lavozdelosquesobran.cl/']

    rules = ()

    def parse_item(self, response):
        pass


class TheClinicSpider(CrawlSpider):
    name = 'theclinic'
    allowed_domains = ['theclinic.cl']
    start_urls = ['https://www.theclinic.cl/']

    rules = ()

    def parse_item(self, response):
        pass
