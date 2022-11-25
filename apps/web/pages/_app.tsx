import type { AppProps } from 'next/app';
import { ChakraProvider } from '@chakra-ui/react';
import type { ReactElement, ReactNode } from 'react';
import type { NextPage } from 'next';
import theme from '@utils/theme';
import { SessionProvider } from 'next-auth/react';

export type NextPageWithLayout<P = Record<string, unknown>, IP = P> = NextPage<
  P,
  IP
> & {
  getLayout?: (page: ReactElement) => ReactNode;
};

type AppPropsWithLayout = AppProps & {
  Component: NextPageWithLayout;
};

export default function App({
  Component,
  pageProps: { session, ...pageProps },
}: AppPropsWithLayout) {
  // eslint-disable-next-line arrow-parens
  const getLayout = Component.getLayout ?? (page => page);

  return (
    <>
      <SessionProvider session={session}>
        <ChakraProvider theme={theme}>
          {getLayout(<Component {...pageProps} />)}
        </ChakraProvider>
      </SessionProvider>
    </>
  );
}
