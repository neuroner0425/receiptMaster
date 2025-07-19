from icrawler.builtin import BingImageCrawler, GoogleImageCrawler
import os

target_name = '종이 영수증'
save_dir = os.path.join('..', 'receipt_data', 'crawl')
min_count = 200

os.makedirs(save_dir, exist_ok=True)

current_count = len([
    f for f in os.listdir(save_dir)
    if os.path.isfile(os.path.join(save_dir, f))
])

need = min_count - current_count

if need <= 0:
    print(f"[{target_name}] 이미 {min_count}장 이상 존재합니다. 크롤링 필요 없음.")
else:
    print(f"[{target_name}] 현재 {current_count}장, {need}장 크롤링 시작...")

    per_source = need // 2
    extra = need % 2

    # Bing 크롤러
    print(f"[{target_name}] Bing에서 {per_source + extra}장 크롤링 중...")
    bing_crawler = BingImageCrawler(storage={'root_dir': save_dir})
    bing_crawler.crawl(keyword=target_name, max_num=per_source + extra)

    # Google 크롤러
    print(f"[{target_name}] Google에서 {per_source}장 크롤링 중...")
    google_crawler = GoogleImageCrawler(storage={'root_dir': save_dir})
    google_crawler.crawl(keyword=target_name, max_num=per_source)

    print(f"[{target_name}] 크롤링 완료.")
