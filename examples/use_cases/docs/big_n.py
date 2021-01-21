from kiri import ChunkedDocument, ElasticChunkedDocument

TSLA = """Tesla, Inc. (formerly Tesla Motors, Inc.) is an American electric vehicle and clean energy company based in Palo Alto, California. Tesla's current products include electric cars, battery energy storage from home to grid scale, solar panels and solar roof tiles, and related products and services.

Founded in July 2003 as Tesla Motors, the company's name is a tribute to inventor and electrical engineer Nikola Tesla. Elon Musk, who contributed most of the funding in the early days, has served as CEO since 2008. According to Musk, the purpose of Tesla is to help expedite the move to sustainable transport and energy, obtained through electric vehicles and solar power.[9][10]

Tesla ranked as the world's best-selling plug-in and battery electric passenger car manufacturer in 2019, with a market share of 17 percent of the plug-in segment and 23 percent of the battery electric segment. Tesla global vehicle sales were 367,849 units in 2019, a 50 percent increase over the previous year. In 2020, the company surpassed the 1 million mark of electric cars produced.[11] The Model 3 ranks as the world's all-time best-selling plug-in electric car, with more than 500,000 delivered.[12] Through its subsidiary SolarCity, Tesla is also a major installer of solar PV systems in the United States, and is one of the largest global supplier of battery energy storage systems, from home-scale to grid-scale. Tesla installed some of the largest battery storage plants in the world and supplied 1.65 GWh of battery storage in 2019.

Tesla has been the subject of numerous lawsuits and controversies, arising from the statements and the conduct of CEO Elon Musk, allegations of whistleblower retaliation, alleged worker rights violations, and allegedly unresolved and dangerous technical problems with their products. Tesla is also one of the most shorted companies in history."""

AAPL = """Apple Inc. is an American multinational technology company headquartered in Cupertino, California, that designs, develops and sells consumer electronics, computer software, and online services. It is considered one of the Big Five companies in the U.S. information technology industry, along with Amazon, Google, Microsoft, and Facebook.[8][9][10]

The company's hardware products include the iPhone smartphone, the iPad tablet computer, the Mac personal computer, the iPod portable media player, the Apple Watch smartwatch, the Apple TV digital media player, the AirPods wireless earbuds and the HomePod smart speaker line. Apple's software includes macOS, iOS, iPadOS, watchOS, and tvOS operating systems, the iTunes media player, the Safari web browser, the Shazam music identifier and the iLife and iWork creativity and productivity suites, as well as professional applications like Final Cut Pro, Logic Pro, and Xcode. Its online services include the iTunes Store, the iOS App Store, Mac App Store, Apple Arcade, Apple Music, Apple TV+, iMessage, and iCloud. Other services include Apple Store, Genius Bar, AppleCare, Apple Pay, Apple Pay Cash, and Apple Card.

Apple was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in April 1976 to develop and sell Wozniak's Apple I personal computer, though Wayne sold his share back within 12 days. It was incorporated as Apple Computer, Inc., in January 1977, and sales of its computers, including the Apple II, grew quickly. Within a few years, Jobs and Wozniak had hired a staff of computer designers and had a production line. Apple went public in 1980 to instant financial success. Over the next few years, Apple shipped new computers featuring innovative graphical user interfaces, such as the original Macintosh in 1984, and Apple's marketing advertisements for its products received widespread critical acclaim. However, the high price of its products and limited application library caused problems, as did power struggles between executives. In 1985, Wozniak departed Apple amicably and remained an honorary employee,[11] while Jobs and others resigned to found NeXT.[12]

As the market for personal computers expanded and evolved through the 1990s, Apple lost market share to the lower-priced duopoly of Microsoft Windows on Intel PC clones. The board recruited CEO Gil Amelio to what would be a 500-day charge for him to rehabilitate the financially troubled company—reshaping it with layoffs, executive restructuring, and product focus. In 1997, he led Apple to buy NeXT, solving the desperately failed operating system strategy and bringing Jobs back. Jobs regained leadership status, becoming CEO in 2000. Apple swiftly returned to profitability under the revitalizing Think different campaign, as he rebuilt Apple's status by launching the iMac in 1998, opening the retail chain of Apple Stores in 2001, and acquiring numerous companies to broaden the software portfolio. In January 2007, Jobs renamed the company Apple Inc., reflecting its shifted focus toward consumer electronics, and launched the iPhone to great critical acclaim and financial success. In August 2011, Jobs resigned as CEO due to health complications, and Tim Cook became the new CEO. Two months later, Jobs died, marking the end of an era for the company. In June 2019, Jony Ive, Apple's CDO, left the company to start his own firm, but stated he would work with Apple as its primary client.

Apple's worldwide annual revenue totaled $274.5 billion for the 2020 fiscal year. Apple is the world's largest technology company by revenue and one of the world's most valuable companies. It is also the world's third-largest mobile phone manufacturer after Samsung and Huawei.[13] In August 2018, Apple became the first publicly traded U.S. company to be valued at over $1 trillion[14][15] and just two years later in August 2020 became the first $2 trillion U.S. company.[16][17] The company employs 137,000 full-time employees[18] and maintains 510 retail stores in 25 countries as of 2020.[19] It operates the iTunes Store, which is the world's largest music retailer. As of January 2020, more than 1.5 billion Apple products are actively in use worldwide.[20] The company also has a high level of brand loyalty and is ranked as the world's most valuable brand. However, Apple receives significant criticism regarding the labor practices of its contractors, its environmental practices and unethical business practices, including anti-competitive behavior, as well as the origins of source materials."""

MSFT = """Microsoft Corporation is an American multinational technology company with headquarters in Redmond, Washington. It develops, manufactures, licenses, supports, and sells computer software, consumer electronics, personal computers, and related services. Its best known software products are the Microsoft Windows line of operating systems, the Microsoft Office suite, and the Internet Explorer and Edge web browsers. Its flagship hardware products are the Xbox video game consoles and the Microsoft Surface lineup of touchscreen personal computers. Microsoft ranked No. 21 in the 2020 Fortune 500 rankings of the largest United States corporations by total revenue;[3] it was the world's largest software maker by revenue as of 2016.[4] It is considered one of the Big Five companies in the U.S. information technology industry, along with Google, Apple, Amazon, and Facebook.

Microsoft (the word being a portmanteau of "microcomputer software"[5]) was founded by Bill Gates and Paul Allen on April 4, 1975, to develop and sell BASIC interpreters for the Altair 8800. It rose to dominate the personal computer operating system market with MS-DOS in the mid-1980s, followed by Microsoft Windows. The company's 1986 initial public offering (IPO), and subsequent rise in its share price, created three billionaires and an estimated 12,000 millionaires among Microsoft employees. Since the 1990s, it has increasingly diversified from the operating system market and has made a number of corporate acquisitions, their largest being the acquisition of LinkedIn for $26.2 billion in December 2016,[6] followed by their acquisition of Skype Technologies for $8.5 billion in May 2011.[7]

As of 2015, Microsoft is market-dominant in the IBM PC compatible operating system market and the office software suite market, although it has lost the majority of the overall operating system market to Android.[8] The company also produces a wide range of other consumer and enterprise software for desktops, laptops, tabs, gadgets, and servers, including Internet search (with Bing), the digital services market (through MSN), mixed reality (HoloLens), cloud computing (Azure), and software development (Visual Studio).

Steve Ballmer replaced Gates as CEO in 2000, and later envisioned a "devices and services" strategy.[9] This unfolded with Microsoft acquiring Danger Inc. in 2008,[10] entering the personal computer production market for the first time in June 2012 with the launch of the Microsoft Surface line of tablet computers, and later forming Microsoft Mobile through the acquisition of Nokia's devices and services division. Since Satya Nadella took over as CEO in 2014, the company has scaled back on hardware and has instead focused on cloud computing, a move that helped the company's shares reach its highest value since December 1999.[11][12]

Earlier dethroned by Apple in 2010, in 2018 Microsoft reclaimed its position as the most valuable publicly traded company in the world.[13] In April 2019, Microsoft reached the trillion-dollar market cap, becoming the third U.S. public company to be valued at over $1 trillion after Apple and Amazon respectively.[14]"""

AMZN = """Amazon.com, Inc.[7], is an American multinational technology company based in Seattle, Washington, which focuses on e-commerce, cloud computing, digital streaming, and artificial intelligence. It is considered one of the Big Five companies in the U.S. information technology industry, along with Google, Apple, Microsoft, and Facebook.[8][9][10][11] The company has been referred to as "one of the most influential economic and cultural forces in the world", as well as the world's most valuable brand.[12][13]

Amazon was founded by Jeff Bezos in Bellevue, Washington, on July 5, 1994. The company started as an online marketplace for books but expanded to sell electronics, software, video games, apparel, furniture, food, toys, and jewelry. In 2015, Amazon surpassed Walmart as the most valuable retailer in the United States by market capitalization.[14] In 2017, Amazon acquired Whole Foods Market for US$13.4 billion, which substantially increased its footprint as a physical retailer.[15] In 2018, Bezos announced that its two-day delivery service, Amazon Prime, had surpassed 100 million subscribers worldwide.[16][17]

Amazon is known for its disruption of well-established industries through technological innovation and mass scale.[18][19][20] It is the world's largest online marketplace, AI assistant provider, live-streaming platform and cloud computing platform[21] as measured by revenue and market capitalization.[22] Amazon is the largest Internet company by revenue in the world.[23] It is the second largest private employer in the United States[24] and one of the world's most valuable companies.

Amazon distributes downloads and streaming of video, music, and audiobooks through its Prime Video, Amazon Music, Twitch, and Audible subsidiaries. Amazon also has a publishing arm, Amazon Publishing, a film and television studio, Amazon Studios, and a cloud computing subsidiary, Amazon Web Services. It produces consumer electronics including Kindle e-readers, Fire tablets, Fire TV, and Echo devices. Its acquisitions over the years include Ring, Twitch, Whole Foods Market, and IMDb. The company has been criticized for various practices including technological surveillance overreach,[25] a hyper-competitive and demanding work culture,[26] tax avoidance,[27] and for being anti-competitive."""

TSLA_doc = ChunkedDocument(TSLA, chunking_level=5)
AAPL_doc = ChunkedDocument(AAPL, chunking_level=5)
MSFT_doc = ChunkedDocument(MSFT, chunking_level=5)
AMZN_doc = ChunkedDocument(AMZN, chunking_level=5)

TSLA_edoc = ElasticChunkedDocument(TSLA, chunking_level=5)
AAPL_edoc = ElasticChunkedDocument(AAPL, chunking_level=5)
MSFT_edoc = ElasticChunkedDocument(MSFT, chunking_level=5)
AMZN_edoc = ElasticChunkedDocument(AMZN, chunking_level=5)

big_n_docs = {"memory": [TSLA_doc, AAPL_doc, MSFT_doc, AMZN_doc],
              "elastic": [TSLA_edoc, AAPL_edoc, MSFT_edoc, AMZN_edoc]}