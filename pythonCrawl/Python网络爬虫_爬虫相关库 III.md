# 爬虫框架

## Scrapy

### 1. 官方文档

https://doc.scrapy.org

### 2. 特点

高性能持久化存储、异步数据下载、高性能数据解析、分布式

### 3. 安装

```shell
pip install scrapy
```



### 4. 使用

#### 开启项目

- 项目开启

```shell
scrapy startproject projectname
```

项目下有多个文件：

spiders 目录：存放爬虫文件

items.py：定义要存储数据的字段

middlewares.py：就是中间件，可以做一些在爬虫过程中想干的事情，比如爬虫在响应的时候你可以做一些操作

pipelines.py：定义一些存储信息的文件，比如我们要连接 MySQL或者 MongoDB 就可以在这里定义

settings.py：定义各种配置，比如配置请求头信息等

- 创建spider.py

```shell
cd projectname
scrapy genspider spiderName www.xxx.com
```

- 设置

```shell
### setting.py
ROBOTSTXT_OBEY = FALSE    # 学习阶段可以调为False
USER_AGENT = 'xxx'
LOG_LEVEL = 'ERROR'    # 日志级别
FEED_EXPORT_ENCODING = 'utf-8' # 中文乱码
```

- 执行爬虫

```shell
scrapy crawl spiderName
```

#### 数据解析

```python
### spider.py
import scrapy

class QiushiSpider(scrapy.Spider):
    name = "qiushibaike"  # 爬虫源文件的唯一标识符
    # allowed_domains = ['www.xxx.com']  # 允许域名（限定start_urls中可以被请求的url）
    start_urls = ['xxx']  # 起始的url列表，自动进行请求

    def parse(self, response):
        # 解析数据
        content_left_div = response.xpath('//*[@id="content-left"]')
        content_list_div = content_left_div.xpath('./div')
		
        all_data = []
        for content_div in content_list_div:
            author = content_div.xpath('./div/a[2]/h2/text()').get()   # 还有extract方法
            content = content_div.xpath('./a/div/span/text()').getall()
            _id = content_div.attrib['id']
            dic = {
                'author': author,
                'content': content,
                'id': _id
            }
            all_data.append(dic)
		return all_data
```

get() 、getall() 方法是新的方法，extract() 、extract_first()方法是旧的方法。extract() 、extract_first()方法取不到就返回None。get() 、getall() 方法取不到就raise一个错误。

#### 持久化存储

- 基于终端的持久化存储——支持csv/xml/json/pickle等文件

```shell
scrapy crawl qiushibaike -o qiushibaike.json
scrapy crawl qiushibaike -o qiushibaike.csv
```

- 基于管道的持久化存储

解析后将数据封装到Item类中，并将item提交给管道

```python
### spider.py
from qiushibaike.Items import QiushibaikeItem
def parse(self, response):
    # 解析数据
    content_left_div = response.xpath('//*[@id="content-left"]')
    content_list_div = content_left_div.xpath('./div')
    
    for content_div in content_list_div:
        item = QiushibaikeItem()
        item['author'] = content_div.xpath('./div/a[2]/h2/text()').get()
        item['content'] = content_div.xpath('./a/div/span/text()').getall()
        item['id'] = content_div.attrib['id']
        
        yield item # 提交给管道
```

```python
### items.py
import scrapy

class QiushibaikeItem(scrapy.Item):
    # 定义所有爬取的字段
    author = scrapy.Field()
    content = scrapy.Field()
    _id = scrapy.Field()
```

在管道中开启持久化存储

```python
### pipelines.py
class QiushibaikePipeline(object):
    fp = None
    def open_spider(self, spider):
        # 重写父类方法，仅在爬虫开始时调用一次
        self.fp = open('./a.txt', 'w', encoding='utf-8')
    
    def process_item(self, item, spider):
        author = item['author']
        content = item['content']
        _id = item['id']
        self.fp.write(author + content + _id)
        return item  # 可以传递给下一个即将被执行的管道类
    
    def close_spider(self, spider):
        # 重写父类方法，仅在爬虫开始时调用一次
        self.fp.close()
```

在配置文件中开启管道

```shell
### setting.py
ITEM_PIPELINES = {
	’qiushibaike.pipelines.QiushibaikePipeline‘: 300,
}  # 300表示优先级，数值越小优先级越高
```

若要存储到多个载体中，可以定义多个管道类，并在设置中设置（可以设置不同优先级）



### 5. 五大核心组件

- 组件：引擎、spider、调度器、下载器、管道

<img src="C:\Users\27110\AppData\Roaming\Typora\typora-user-images\image-20220528164236561.png" alt="image-20220528164236561" style="zoom:50%;" />

- 爬虫流程

spider将启动对象（url）发送给引擎

引擎将启动对象发给调度器

调度器首先对启动对象进行去重，然后加入队列，返回引擎

引擎将信息传递给下载器进行下载获得response返回引擎

引擎将response传递给spider解析后再传回引擎

引擎将解析后的数据传给管道进行永久化存储



### 6. 高级用法

#### 请求传参

对详情页发送请求

```python
### spider.py
import scrapy
from qiushibaike.Items import QiushibaikeItem

class QiushiSpider(scrapy.Spider):
    name = "qiushibaike"  # 爬虫源文件的唯一标识符
    start_urls = ['xxx']  # 起始的url列表，自动进行请求
    url = 'xxxpage=%d'
    page_num = 2

	def parse_detail(self, response):
		item = response.meta['item']
        item['xxx'] = content_div.xpath('./div/a[2]/h2/text()').get()
        yield item

    def parse(self, response):
        content_list_div = content_left_div.xpath('./div')		
        for content_div in content_list_div:
        	item = QiushibaikeItem()
        	item['xxx'] = content_div.xpath('./div/a[2]/h2/text()').get()
            detail_url = content_div.xpath('./div/a[2]/h2/text()').get()
			yield scrapy.Request(detail_url, callback=self.parse_detail, meta={'item': item})  # 通过meta进行请求传参
        
        # 分页操作
        if self.page_num <=3:
        	new_url = format(self.url % self.page_num)
            self.page_num += 1
            yield scrapy.Request(new_url, callback=self.parse)
```

#### 图片爬取

ImagesPipeline

将src属性值传递给Pipeline

图片懒加载问题

```python
class ImgSpider(scrapy.Spider):
    name = 'img'
    start_urls = ['https://sc.chinaz.com/tupian/']

    def parse(self, response):
        div_list = response.xpath('//div[@id="container"]/div')
        for div in div_list:
            src = div.xpath('./div/a/img/@src2').extract_first()    # 使用伪属性
            # print(src)

            item = ZhanzhangproItem()
            item['src'] = 'https:' + src
            print(item['src'])
            yield item
```

```python
class imagesPipeline(ImagesPipeline):
    def get_media_request(self, item, info):
        """根据图片地址进行数据请求"""
        yield scrapy.Request(item['src'])

    def file_path(self, request, response=None, info=None):
        """定义图片保存路径"""
        imgName = request.url.split('/')[-1]  # 图片名称（图片保存路径在setting.py中定义）
        return imgName

    def item_completed(self, results, item, info):
        """返回下一个即将被执行的管道类"""
        return item
```

```python
### setting.py
ITEM_PIPELINES = {
   'zhanzhangPro.pipelines.imagesPipeline': 300,
}
# 图片存储目录
IMAGES_STORE = './imgs'
```

#### 中间件

下载中间件

拦截请求（UA伪装、代理IP）

拦截响应（修改响应数据）

```python
### middlewares.py

class ZhanzhangproDownloaderMiddleware:
    user_agent_list = []
    PROXY_http = []
    PORXY_https = []
    
    # 拦截请求
    def process_request(self, request, spider):
        # UA伪装（设置为不同UA）
        request.header['User-Agent'] = ramdom.choice(self.user_agent_list)
        return None

    def process_response(self, request, response, spider):
        # Called with the response returned from the downloader.

        # Must either;
        # - return a Response object
        # - return a Request object
        # - or raise IgnoreRequest
        return response

    def process_exception(self, request, exception, spider):
        # 代理IP
        if request.url.split(':')[0] = 'http':
            request['proxy'] = 'http' + random.choice(self.PROXY_http)
        else:
            request['proxy'] = 'https' + random.choice(self.PROXY_https)
    	return request  # 重新请求对象
      
```

```python
### setting.py中开启中间件
DOWNLOADER_MIDDLEWARES = {
   'zhanzhangPro.middlewares.ZhanzhangproDownloaderMiddleware': 543,
}
```

#### CrawlSpider全站数据爬取

- 创建爬虫文件

```
scrapy genspider -t crawl xxx www.xxx.com
```

- 创建链接提取器和规则提取器

```python
from distutils.ccompiler import new_compiler
import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import CrawlSpider, Rule
from sunPro.items import SunproItem, DetailItem

class SunSpider(CrawlSpider):
    name = 'sun'
    start_urls = ['https://wz.sun0769.com/political/index/politicsNewest?id=1&page=1']

    # 链接提取器：指定链接提取规则（allow为页码图标正则）
    link = LinkExtractor(allow=r'id=1&page=\d+')
    link_detail = LinkExtractor(allow=r'question/\d+/\d+/.shtml')

    rules = (
        # 规则解析器
        Rule(link, callback='parse_item', follow=True),
        Rule(link_detail, callback='parse_detail')
    )

    def parse_item(self, response):
        # xpath表达式中不能出现tbody标签
        tr_list = response.xpath('//*[@id="morelist"]/div/table[2]//tr/td/table//tr')
        for tr in tr_list:
            new_num = tr.xpath('./td[1]/text()').extract_first()
            new_title = tr.xpath('./td[2]/a[2]/@title').extract_first()
            item = SunproItem()
            item['new_title'] = new_title
            item['new_num'] = new_num
            yield item

    def parse_detail(self, response):
        new_id = response.xpath('xxx').extract_first()
        new_content = response.xpath('xxx').extract()
        new_content = ''.join(new_content)
        item = DetailItem()
        item['new_content'] = new_content
        item['new_id'] = new_id
        yield item
```



### 7. 分布式爬虫

#### 分布式

原生的scrapy不能实现分布式爬虫：调度器和管道不能被分布式机群共享

scrapy-redis：可以提供能能被分布式机群共享的调度器和管道

#### 使用

- 创建工程及爬虫文件

```shell
scrapy startproject xxx
scrapy genspider -t crawl xxx xxx.com
```

- 修改爬虫文件

```python
from scrapy_redis.spiders import RedisCrawlSpider
import scrapy
from scrapy.linkextractors import LinkExtractor
from scrapy.spiders import Rule
from sunPro.items import SunproItem, DetailItem

class SunSpider(RedisCrawlSpider):
    name = 'sun'
    # start_urls = ['https://wz.sun0769.com/political/index/politicsNewest?id=1&page=1']
    redis_key = 'abc'

    # 链接提取器：指定链接提取规则（allow为页码图标正则）
    link = LinkExtractor(allow=r'id=1&page=\d+')
    link_detail = LinkExtractor(allow=r'question/\d+/\d+/.shtml')

    rules = (
        # 规则解析器
        Rule(link, callback='parse_item', follow=True),
        Rule(link_detail, callback='parse_detail')
    )

    def parse_item(self, response):
        # xpath表达式中不能出现tbody标签
        tr_list = response.xpath('//*[@id="morelist"]/div/table[2]//tr/td/table//tr')
        for tr in tr_list:
            new_num = tr.xpath('./td[1]/text()').extract_first()
            new_title = tr.xpath('./td[2]/a[2]/@title').extract_first()
            item = SunproItem()
            item['new_title'] = new_title
            item['new_num'] = new_num
            yield item

    def parse_detail(self, response):
        new_id = response.xpath('xxx').extract_first()
        new_content = response.xpath('xxx').extract()
        new_content = ''.join(new_content)
        item = DetailItem()
        item['new_content'] = new_content
        item['new_id'] = new_id
        yield item


```

- 修改配置文件

指定可以被共享的管道和调度器

```python
ITEM_PIPELINES = {
    'scrapy_redis.pipelines.RedisPipeline': 400
}
DUPEFILTER_CLASS = "scrapy_redis.duperfilter.RFPDupeFilter"  # 去重容器配置
SCHEDULER = "scrapy_redis.scheduler.Scheduler"  # 调度器
SCHEDULER_PERSIST = True

REDIS_HOST = 'xxx'
REDIS_PORT = xxxx
```

- 配置redis的配置文件

```shell
## redis.conf（linux）or redis_windows.conf（windows）
bind 127.0.0.1   # 注释改行
protect-mode yes  # 将保护模式由yes改为no
```

开启redis服务

启动redis客户端

- 执行工程

```shell
scrapy runspider xxx.py
```

- 多台电脑可以向redis调度器中放入起始url

```shell
lpush abc start_url.com
```

- 爬到的数据最终存储在proName: items这个数据结构中



### 8. 增量式爬虫

主程序与CrawlSpider类似，但在发送请求前与request过的网页url进行匹配，若已请求过则不再请求