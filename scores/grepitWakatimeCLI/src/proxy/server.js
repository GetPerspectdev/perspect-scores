/* eslint-disable unicorn/no-process-exit */
/* eslint-disable no-process-exit */
/* eslint-disable unicorn/prefer-module */
const path = require('node:path')
const http = require('node:http')
const https = require('node:https')
const {homedir} = require('node:os')
const pino = require('pino')
const PouchDB = require('pouchdb')
const {removeApiUrl} = require('./server-utils.js')

// Logging
const transport = pino.transport({
  target: 'pino/file',
  options: {destination: `${homedir()}/.sebastian.log`},
})

const logger = pino(transport)

// Cleanup function
function cleanup() {
  removeApiUrl()
  logger.info('running clean up')
  process.exit()
}

const runServer = () => {
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [_, __, username, dataDir = __dirname, port = 8888, sync = true, remoteUrl = 'https://admin:couchdb@lil-cuz.fly.dev/'] = process.argv

  logger.info(dataDir)

  // PouchDB/CounchDB
  const dbPath = path.join(dataDir, 'heartbeat')
  const db = new PouchDB(dbPath)

  // WakaTime API
  const wakatime = 'api.wakatime.com'

  let remoteDB = null
  if (username && sync) {
    const url = `${remoteUrl}${username}`
    remoteDB = new PouchDB(url)
    setInterval(async () => {
      try {
        if (remoteDB) {
          await db.replicate.to(remoteDB)
          logger.info({msg: 'Replication Complete'})
        }
      } catch (error) {
        logger.info(new Error(error))
      }
    }, 30_000)
  }

  const server = http.createServer(async function (req, res) {
    if (req.method === 'OPTIONS') {
      const headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': req.headers['access-control-request-headers'],
        'Access-Control-Allow-Methods': 'OPTIONS, POST, GET, PUT',
        'Access-Control-Max-Age': 2_592_000,
      }
      res.writeHead(204, headers)
      res.end()
      return
    }

    if (req.url === '/') {
      db.info().then(function (info) {
        res.statusCode = 200
        res.end(`You are running GrepIt. Local DB Status: ${JSON.stringify(info)}`)
      }).catch(function (error) {
        res.end(error.toString())
      })
      return
    }

    delete req.headers['accept-encoding']
    const headers = {...req.headers, host: wakatime}
    const method = req.method
    const options = {
      host: wakatime,
      port: 443,
      path: `/api/v1${req.url}`,
      method,
      headers,
    }

    logger.info({method, url: options.path, msg: 'Requesting'})

    const request = https.request(options, response => {
      res.statusCode = response.statusCode
      response.on('data', chunk => {
        res.write(chunk)
      })

      response.on('end', () => {
        logger.info({msg: 'Recieved Data'})
        res.end()
      })
    })

    request.on('error', e => {
      console.error(e.message)
    })

    if (method === 'POST' || method === 'PUT') {
      const body = []
      req.on('data', chunk => {
        if (chunk === undefined || !chunk) return
        body.push(chunk.toString('utf8'))
      })
      req.on('end', async () => {
        const postBody = body.join('')
        try {
        // Write to a database
          await db.post({dataSent: JSON.parse(postBody)})
          logger.info({msg: 'Data Saved'})
        } catch (error) {
          logger.info(new Error(error))
        }

        request.write(postBody)
        request.end()
      })
    } else {
      request.end()
    }
  })

  server.listen(port, () => {
    logger.info('Listening on http://localhost:%s...', port)
  })

  server.on('error', e => {
    logger.info(e)
  })

  // Signal handling
  process.on('SIGINT', cleanup)
  process.on('SIGTERM', cleanup)
  process.on('SIGQUIT', cleanup)
}

runServer()
