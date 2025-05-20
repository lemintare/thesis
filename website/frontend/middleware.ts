// middleware.ts
import { NextResponse } from 'next/server'
import type { NextRequest } from 'next/server'

export const config = {
  matcher: '/dashboard/:path*',
}

export async function middleware(request: NextRequest) {
  const token = request.cookies.get('auth_token')?.value
  if (!token) {
    return NextResponse.redirect(new URL('/', request.url))
  }

  const cookieHeader = request.headers.get('cookie') || ''

  const verifyRes = await fetch('http://localhost:8000/auth/verify-token', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'cookie': cookieHeader,
    },
  })

  let data: any = null
  try {
    data = await verifyRes.json()
  } catch {
    data = null
  }

  console.log('Verify status:', verifyRes.status, 'body:', data)

  const isValid =
    (typeof data === 'boolean' && data === true)
    || (typeof data === 'object' && data?.valid === true)

  if (!verifyRes.ok || !isValid) {
    return NextResponse.redirect(new URL('/', request.url))
  }

  return NextResponse.next()
}
