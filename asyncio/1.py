import asyncio

async def mcm(inrange, div):
    print("Finding nums in range {} divisible by {}".format(inrange, div)+"\n")
    loc = []
    for i in range(inrange):
        if i % div == 0:
            loc.append(i)
        if i % 50000 == 0:
            await asyncio.sleep(0.00001)
    print("Done w/ nums in range {} divisible by {}".format(inrange, div)+"\n")
    return loc

async def main():
    div1 = loop.create_task(mcm(508000, 34113))
    div2 = loop.create_task(mcm(100052, 3210))
    div3 = loop.create_task(mcm(500, 3))
    await asyncio.wait([div1, div2, div3])
    return div1, div2, div3
    
if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()      #Inicializa el loop
        div1, div2, div3 = loop.run_until_complete(main())
        print(div1.result())
    except Exception as e:
        pass
    finally:
        loop.close()