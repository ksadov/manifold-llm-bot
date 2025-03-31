from typing import Optional, List, Dict, Any, Union
from enum import Enum
from pydantic import BaseModel


class OutcomeType(str, Enum):
    BINARY = "BINARY"
    FREE_RESPONSE = "FREE_RESPONSE"
    MULTIPLE_CHOICE = "MULTIPLE_CHOICE"
    NUMERIC = "NUMERIC"
    PSEUDO_NUMERIC = "PSEUDO_NUMERIC"
    BOUNTIED_QUESTION = "BOUNTIED_QUESTION"
    POLL = "POLL"
    MULTI_NUMERIC = "MULTI_NUMERIC"
    DATE = "DATE"


class AddAnswersMode(str, Enum):
    ANYONE = "ANYONE"
    ONLYCREATOR = "ONLY_CREATOR"
    DISABLED = "DISABLED"


class Option(BaseModel):
    text: str
    votes: int


class Fees(BaseModel):
    platformFee: float
    liquidityFee: float
    creatorFee: float


class Fill(BaseModel):
    amount: int | float
    matchedBetId: Optional[str]
    shares: float
    timestamp: int


class Bet(BaseModel):
    amount: int | float
    contractId: str
    orderAmount: int | float
    shares: float
    isFilled: bool
    isCancelled: bool
    fills: List[Fill]
    outcome: str
    probBefore: float
    probAfter: float
    loanAmount: int | float
    createdTime: int
    isRedemption: bool
    visibility: str
    betId: str
    fees: Fees
    limitProb: Optional[float] = None
    expiresMillisAfter: Optional[int] = None
    expiresAt: Optional[int] = None


class LiteMarket(BaseModel):
    id: str
    creatorId: str
    creatorUsername: str
    creatorName: str
    createdTime: int
    question: str
    url: str
    outcomeType: OutcomeType
    mechanism: str
    totalLiquidity: Optional[float] = None
    volume: float
    volume24Hours: float
    isResolved: bool
    uniqueBettorCount: int
    creatorAvatarUrl: Optional[str] = None
    closeTime: Optional[int] = None
    p: Optional[float] = None
    probability: Optional[float] = None
    pool: Optional[Dict[str, Any]] = None
    value: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    isLogScale: Optional[bool] = None
    resolutionTime: Optional[int] = None
    resolution: Optional[str] = None
    resolutionProbability: Optional[float] = None
    lastUpdatedTime: Optional[int] = None
    lastBetTime: Optional[int] = None
    marketTier: Optional[str] = None
    siblingContractId: Optional[str] = None
    slug: Optional[str] = None
    token: Optional[str] = None
    lastCommentTime: Optional[int] = None
    resolverId: Optional[str] = None
    comments: Optional[List[str]] = None


class FullMarket(LiteMarket):
    description: Dict[str, Any]
    textDescription: str
    answers: Optional[List[dict]] = None
    shouldAnswersSumToOne: Optional[bool] = None
    addAnswersMode: Optional[AddAnswersMode] = None
    options: Optional[List[Option]] = None
    totalBounty: Optional[float] = None
    bountyLeft: Optional[float] = None
    coverImageUrl: Optional[str] = None
    groupSlugs: List[str] = []


class User(BaseModel):
    id: str
    createdTime: int
    name: str
    username: str
    url: str
    balance: float
    totalDeposits: float
    lastBetTime: Optional[int] = None
    currentBettingStreak: Optional[int] = None
    avatarUrl: Optional[str] = None
    bio: Optional[str] = None
    bannerUrl: Optional[str] = None
    website: Optional[str] = None
    twitterHandle: Optional[str] = None
    discordHandle: Optional[str] = None
    isBot: Optional[bool] = None
    isAdmin: Optional[bool] = None
    isTrustworthy: Optional[bool] = None
    isBannedFromPosting: Optional[bool] = None
    userDeleted: Optional[bool] = None
    verifiedPhone: Optional[bool] = None
    creatorTraders: Optional[dict] = None
    signupBonusPaid: Optional[bool] = None
    isBannedFromMana: Optional[bool] = None
    nextLoanCached: Optional[int] = None
    hasSeenLoanModal: Optional[bool] = None
    isAdvancedTrader: Optional[bool] = None
    kycDocumentStatus: Optional[str] = None
    optOutBetWarnings: Optional[bool] = None
    shouldShowWelcome: Optional[bool] = None
    streakForgiveness: Optional[int] = None
    followerCountCached: Optional[int] = None
    isBannedFromSweepcash: Optional[bool] = None
    cashBalance: Optional[float] = None
    spiceBalance: Optional[float] = None
    totalCashDeposits: Optional[float] = None
