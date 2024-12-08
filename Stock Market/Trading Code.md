# Code by Friend for Indicator & Alerts

```pinescript
// This source code is subject to the terms of the Mozilla Public License 2.0 at https://mozilla.org/MPL/2.0/

// © Sumit-Li

  

//@version=5

indicator("Five Star strategy(APPA)")

len = input.int(5, minval=1, title="Length")

src = input(close, title="Source")

offset = input.int(title="Offset", defval=0, minval=-500, maxval=500)

out = ta.ema(src, len)

plot(out, title="EMA", color=color.blue, offset=offset)

  

ma(source, length, type) =>

    switch type

        "SMA" => ta.sma(source, length)

        "EMA" => ta.ema(source, length)

        "SMMA (RMA)" => ta.rma(source, length)

        "WMA" => ta.wma(source, length)

        "VWMA" => ta.vwma(source, length)

  

typeMA = input.string(title = "Method", defval = "EMA", options=["SMA", "EMA", "SMMA (RMA)", "WMA", "VWMA"], group="Smoothing")

smoothingLength = input.int(title = "Length", defval = 5, minval = 1, maxval = 100, group="Smoothing")

  

smoothingLine = ma(out, smoothingLength, typeMA)

plot(smoothingLine, title="Smoothing Line", color=#f37f20, offset=offset, display=display.none)

  

o = open

h = high

l  = low

c = close

if (out>o and out>h and out>l and out>c)

    alert("BUY alert when candle HIGH is break", alert.freq_once_per_bar_close)

else if(out<o and out<h and out<l and out<c)

    alert("SELL alert when candle LOW is break", alert.freq_once_per_bar_close)
```