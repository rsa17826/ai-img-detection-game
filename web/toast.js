// ==UserScript==
// @name         lib:toast
// @version      3
// @description  allows showing toast message
// @license      GPLv3
// @run-at       document-start
// @author       rssaromeo
// @match        *://*/*
// @tag          lib
// @exclude      /livereload.net\/files\/ffopen\/index.html$/
// @icon         data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEgAAABICAMAAABiM0N1AAAAAXNSR0IB2cksfwAAAAlwSFlzAAAOxAAADsQBlSsOGwAAAHJQTFRFAAAAEIijAo2yAI60BYyuF4WaFIifAY6zBI2wB4usGIaZEYigIoiZCIyrE4igG4iYD4mjEomhFoedCoqpDIqnDomlBYyvE4efEYmiDYqlA42xBoytD4mkCYqqGYSUFYidC4qoC4upAo6yCoupDYqmCYur4zowOQAAACZ0Uk5TAO////9vr////1+/D/+/L+/Pf/////+f3///////H4////////+5G91rAAACgUlEQVR4nM2Y22KjIBCGidg1264liZqDadK03X3/V2wNKHMC7MpF/xthHD5mgERAqZhWhfYqH6K+Qf2qNNf625hCoFj9/gblMUi5q5jLkXLCKudgyiRm0FMK82cWJp1fLbV5VmvJbCIc0GCYaFqqlDJgADdBjncqAXYobm1xh72aFMflbysteFfdy2Yi1XGOm5HGBzQ1dq7TzEoxjeNTjQZb7VA3e1c7+ImgasAgQ9+xusNVNZIo5xmOMgihIS2PbCQIiHEUdTvhxCcS/kPomfFI2zHy2PkWmA6aNatIJpKFJyekyy02xh5Y3DI9T4aOT6VhIUrsNTFp1pf79Z4SIIVDegl6IJO6cHiL/GimIZDhgTu/BlYWCQzHMl0zBWT/T3KAhtxOuUB9FtBrpsz0RV4xsjHmW+UCaffcSy/5viMGer0/6HdFNMZBq/vjJL38H9Dqx4Fuy0Em12DbZy+9pGtiDijbglwAehyj11n0tRD3WUBm+lwulE/8h4BuA+iWAQQnteg2Xm63WQLTpnMnpjdge0Mgu/GRPsV4xdjQ94Lfi624fabhDkfUqIKNrM64Q837v8yL0prasepCgrtvw1sJpoqanGEX7b5mQboNW8eawXaWXTMfMGxub472hzWzHSn6Sg2G9+6TAyRruE71s+zAzjWaknoyJCQzwxrghH2k5FDT4eqWunuNxyN9QCGcxVod5oADbYnIUkDTGZEf1xDJnSFteQ3KdsT8zYDMQXcHxsevcLH1TrsABzkNPyA/L7b0jg704viMMlpQI96WsHknCt/3YH0kOEo9zcGkwrFK39ck72rmoehmKqo2RKlilzSy/nJKEV45CT38myJp456fezktHjN5aeMAAAAASUVORK5CYII=
// @grant        none
// @namespace https://greasyfork.org/users/1184528
// @downloadURL https://update.greasyfork.org/scripts/491569/lib%3Atoast.user.js
// @updateURL https://update.greasyfork.org/scripts/491569/lib%3Atoast.meta.js
// ==/UserScript==
;(async () => {
  const a = loadlib("allfuncs")
  await a.bodyload()
  var toastbox = a.createelem(document.body, "div", {
    id: "toastbox",
  })
  function toast(msg, color) {
    var x = a.createelem(toastbox, "div", {
      class: "toast",
      backgroundColor:
        {
          red: "rgb(255, 127, 127)",
          yellow: "rgb(255, 246, 127)",
          green: "rgb(127, 255, 138)",
        }[color] ?? color,
      innerHTML: msg,
    })
    setTimeout(() => x.remove(), 2500)
    return x
  }
  loadlib("libloader").savelib("toast", toast)
  a.createelem(document.body, "style", {
    innerHTML: `#toastbox {
    z-index: 2;
    display: flex;
    flex-direction: column;
    width: fit-content;
    height: fit-content;
    background-color: transparent;
    min-height: 100px;
    position: fixed;
    top: 10px;
    right: 10px;
    gap: 20px;
    opacity: 0.7;
    font-size: large;
    font-weight: 150;
    pointer-events: none;
  }

  .toast {
    text-align: center;
    line-height: 40px;
    height: 40px;
    padding: 10px;
    border-radius: 20px;
    min-width: 200px;
    color: #000;
    opacity: 1;
    animation: fade 0.5s 1;
    animation-fill-mode: forwards;
    animation-delay: 2s;
  }
  @keyframes fade {
    0% {
      opacity: 0.7;
    }

    100% {
      opacity: 0;
    }
  }`,
  })
})()
