import scrapy


class KiasuSpider(scrapy.Spider):
    name = 'kiasuparent'

    start_urls = [
        'https://www.kiasuparents.com/kiasu/forum/viewforum.php?f=5',
    ]

    def parse(self, response):
        for topic_list in response.xpath('//ul[has-class("topiclist topics")]'):
            for topic in topic_list.xpath('li/dl/dt'):
                yield {
                    'topic': topic.xpath('div/a/text()').get(),
                }
                yield response.follow(topic.xpath('div/a/@href').get(), \
                    self.parse)

        for post in response.xpath('//div[has-class("page-body-inner")]/div/div[has-class("inner")]'):
            yield {
                'author': post.xpath('//*[has-class("author")]/span/strong/a/text()').get(),
                'content': post.xpath('div[has-class("postbody")]/div/div[has-class("content")]/text()').get(),
            }

        next_page = response.xpath('//li[has-class("arrow next")]/a/@href').get()
        if next_page is not None:
            yield response.follow(next_page, self.parse)
